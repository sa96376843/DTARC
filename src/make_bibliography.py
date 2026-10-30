"""Build references.bib from verified Crossref and official proceedings records."""
from pathlib import Path
import html,json,re

ROOT=Path(__file__).resolve().parents[1]
meta=json.loads((ROOT/"literature"/"crossref_verified.json").read_text())
# Correct two early-access records to their final journal versions.
meta["lu2019"].update({"volume":"31","issue":"12","page":"2346-2363","published":{"date-parts":[[2019,12]]}})
meta["delange2022"].update({"volume":"44","issue":"7","page":"3366-3385","published":{"date-parts":[[2022,7]]}})


def clean(s):
    s=html.unescape(re.sub(r"<[^>]+>","",str(s)))
    return s.replace("&",r"\&").replace("%",r"\%").replace("_",r"\_")


def authors(m):
    return " and ".join(clean(" ".join(filter(None,[a.get("given"),a.get("family")]))) for a in m.get("author",[]))


def bibentry(key,m):
    title=clean(m["title"][0]);year=m["published"]["date-parts"][0][0]
    containers=m.get("container-title",[])
    ctype=m.get("type","")
    if ctype in ["proceedings-article","book-chapter"] or any("Proceedings" in c or "Computer Vision" in c for c in containers):
        kind="inproceedings";field="booktitle";container=clean(containers[-1]) if containers else ""
    else:
        kind="article";field="journal";container=clean(containers[0]) if containers else ""
    fields=[("author",authors(m)),("title",title),(field,container),("year",str(year))]
    for k,bibk in [("volume","volume"),("issue","number"),("page","pages"),("publisher","publisher")]:
        if m.get(k):fields.append((bibk,clean(m[k]).replace("-","--") if k=="page" else clean(m[k])))
    if m.get("DOI"):fields.append(("doi",m["DOI"]))
    if m.get("URL"):fields.append(("url",m["URL"]))
    return "@%s{%s,\n%s\n}\n"%(kind,key,",\n".join(f"  {k} = {{{v}}}" for k,v in fields if v))

text="% Bibliography verified against Crossref metadata and official proceedings.\n"
for key in meta:
    text+=bibentry(key,meta[key])+"\n"
text+=r"""@inproceedings{rolnick2019,
  author = {David Rolnick and Arun Ahuja and Jonathan Schwarz and Timothy Lillicrap and Gregory Wayne},
  title = {Experience Replay for Continual Learning},
  booktitle = {Advances in Neural Information Processing Systems 32},
  year = {2019},
  pages = {348--358},
  url = {https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html}
}

@misc{metada2024,
  author = {Shiluo Huang and Zheng Liu and Ye Deng and Qing Li},
  title = {Incremental Learning of Stock Trends via Meta-Learning with Dynamic Adaptation},
  year = {2024},
  eprint = {2401.03865},
  archiveprefix = {arXiv},
  primaryclass = {cs.CE},
  doi = {10.48550/arXiv.2401.03865},
  url = {https://arxiv.org/abs/2401.03865}
}

@book{lopez2018,
  author = {Marcos L{\'o}pez de Prado},
  title = {Advances in Financial Machine Learning},
  publisher = {John Wiley \& Sons},
  address = {Hoboken, NJ},
  year = {2018},
  isbn = {9781119482086},
  url = {https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086}
}
"""
(ROOT/"manuscript").mkdir(exist_ok=True)
(ROOT/"manuscript"/"references.bib").write_text(text)
print("references",len(meta)+3)
