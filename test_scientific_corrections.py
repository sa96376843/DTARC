"""Synthetic regression fixtures, never manuscript observations."""
from pathlib import Path
import sys, unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import tempfile, torch
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import experiment as ex
from preprocess import add_market_context
from statistical_analysis import weighted_metrics_by_date, metrics, blocks
from economic_evaluation import buy_hold, max_drawdown

class Corrections(unittest.TestCase):
    def test_numerical_and_empty_boundaries(self):
        class IdentityLogits(torch.nn.Module):
            def forward(self,x):return x[:,0]
        np.testing.assert_array_equal(ex.predict(IdentityLogits(),np.array([[-1e4],[1e4]],np.float32)),[0.,1.])
        self.assertEqual(len(ex.predict(IdentityLogits(),np.empty((0,1),np.float32))),0)
        self.assertTrue(torch.isfinite(torch.nn.BCEWithLogitsLoss()(torch.tensor([-1e4,1e4]),torch.tensor([1.,0.]))))
        empty=pd.DataFrame(columns=['market','split','h','method','seed','y','p'])
        self.assertTrue(ex.evaluate(empty).empty)
        one=pd.DataFrame({'market':'CN','split':'test','h':5,'method':'fixture','seed':1,'y':[1,1],'p':[0.,0.]})
        scores=ex.evaluate(one).iloc[0]
        self.assertEqual(scores.mcc,0);self.assertEqual(scores.macro_f1,0);self.assertTrue(np.isnan(scores.auc))
        dates=pd.bdate_range('2024-01-01',periods=20)
        with tempfile.TemporaryDirectory() as temporary,patch.object(ex,'ROOT',Path(temporary)):
            for label,weight in [(0,20),(1,0)]:
                train=pd.DataFrame({'date':dates,'code':'a','market':'fixture','constant':0.,'missing':np.nan,'y_1':label})
                fitted=ex.initial_fit(train,['constant','missing'],1,3,epochs=1,purge_boundary=False)
                self.assertEqual(float(fitted['pos_weight'][0]),weight)
                np.testing.assert_array_equal(fitted['scaler'].scale_,[1.,1.])
                self.assertTrue(all(torch.isfinite(p).all() for p in fitted['model'].parameters()))
            with self.assertRaisesRegex(ValueError,'empty after purging'):
                ex.initial_fit(train.iloc[:1],['constant','missing'],1,3,epochs=1)

    def test_page_hinkley_reset(self):
        detector=ex.PageHinkley(delta=0.,threshold=.5)
        for _ in range(20):self.assertIsNone(detector.update(0.))
        hit=detector.update(1.)
        self.assertAlmostEqual(hit['severity'],(20/21)/(np.sqrt(21)+1e-9),places=12)
        self.assertEqual((detector.n,detector.mean,detector.cum,detector.min_cum),(0,0.,0.,0.))
        self.assertIsNone(detector.update(1.));self.assertEqual(detector.n,1)

    def test_cross_section_and_loo(self):
        d=pd.DataFrame({'date':[1,1,1,2], 'code':['a','b','c','a'],
                        'ret_1':[.1,.2,.3,.4], 'mom_5':[.2,.4,.6,.8]})
        full=add_market_context(d);loo=add_market_context(d,True)
        np.testing.assert_allclose(full.market_ret_1,[.2,.2,.2,.4])
        np.testing.assert_allclose(loo.market_ret_1.iloc[:3],[.25,.2,.15])
        self.assertTrue(np.isnan(loo.market_ret_1.iloc[3]))
        np.testing.assert_allclose(full.relative_ret_5,[-.2,0,.2,0],atol=1e-15)

    def test_unique_replay_all_variants(self):
        for size in [1,32,64,80,512]:
            for positives in [0,1,size//2,size]:
                y=np.zeros(size,int);y[:positives]=1;X=np.arange(size*3).reshape(size,3)
                for variant in ['full','no_anchor','no_diversity','no_typing','no_provenance']:
                    for severity in [.1,1.]:
                        for seed in range(10):
                            ids=ex.dtarc_sample(list(range(size)),set(range(size//4)),X,y,np.ones(5),min(64,size),severity,.5,np.random.default_rng(seed),variant)
                            self.assertEqual(len(ids),min(64,size));self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(ex.farthest_select(np.zeros((2,1)),[0,1],0,np.random.default_rng(1)),[])

    def test_bootstrap_exact_weighted_metrics(self):
        rng=np.random.default_rng(71);pos=np.repeat(np.arange(25),3)
        y=rng.integers(0,2,len(pos));p=np.round(rng.uniform(size=len(pos)),1)
        draws=[blocks(25,5,rng) for _ in range(100)]
        counts=np.array([np.bincount(x,minlength=25) for x in draws],float)
        fast=weighted_metrics_by_date(y,p,pos,counts,chunk=13)
        for i,draw in enumerate(draws):
            idx=np.concatenate([np.flatnonzero(pos==j) for j in draw])
            reference=metrics(y[idx],p[idx])
            for key in reference: self.assertAlmostEqual(fast[key][i],reference[key],places=12)
        oneclass=weighted_metrics_by_date(np.ones(len(pos),int),p,pos,counts)
        self.assertTrue(np.isnan(oneclass['auc']).all())
        np.testing.assert_array_equal(oneclass['mcc'],0)

    def test_buyhold_entry_cost(self):
        dates=pd.bdate_range('2026-01-01',periods=20)
        panel=pd.DataFrame({'date':dates,'code':'a','close':100.})
        curve,result=buy_hold(panel,np.array([dates[0],dates[5]]),10)
        self.assertAlmostEqual(result['total_return'],-.001,places=12)
        self.assertAlmostEqual(result['max_drawdown'],.001,places=12)
        self.assertAlmostEqual(result['annual_return'],.999**(252/result['n_periods'])-1,places=12)
        self.assertAlmostEqual(np.prod(1+curve.net_return)-1,result['total_return'],places=12)
        self.assertEqual(result['n_periods'],len(curve)-1)
        self.assertAlmostEqual(max_drawdown([-.1,.05]),.1)

    def test_validation_cutoff(self):
        d=pd.DataFrame({'split':['validation']*2,'y_5':[1,0],
                        'label_available_5':['2025-12-31','2026-01-02']})
        self.assertEqual(len(ex.eligible_split(d,'validation',5)),1)

    def test_saved_prediction_and_initial_memory_maturity(self):
        dates=pd.bdate_range('2024-12-02',periods=50)
        d=pd.DataFrame({'date':dates,'code':'a','market':'CN','split':['train']*30+['validation']*20,
                        'y_1':np.zeros(50), 'fwd_ret_1':np.zeros(50),'label_available_1':pd.bdate_range(dates[1],periods=50)})
        errors=[];calls=[]
        class Detector:
            def update(self,e): errors.append(e);return None
        def prediction(model,x):
            p=.9 if len(calls)%2 else .1;calls.append(len(x));return np.full(len(x),p)
        cfg={'buffer':32,'replay':8,'adwin_delta':.01,'ph_threshold':5}
        with patch.object(ex,'transform',return_value=np.zeros((50,1),np.float32)),patch.object(ex,'predict',side_effect=prediction),patch.object(ex,'make_detector',return_value=Detector()):
            pred,events,run=ex.run_online('adwinER',d,{'model':None},[],1,11,cfg,'validation')
        self.assertEqual(len(calls),4)  # One original prediction per batch; no re-prediction.
        self.assertEqual(errors,[0.]*5+[1.]*5+[0.]*5)
        self.assertEqual(len(pred),20)

if __name__=='__main__':unittest.main(verbosity=2)
