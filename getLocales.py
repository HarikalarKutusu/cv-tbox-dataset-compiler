import json, sys
from typing import Literal
from typing import Any
from urllib.request import urlopen
from const import CV_VERSIONS
from const import CV_DATES
from const import CV_DATASET_BASE_URL

def is_version_valid(ver) -> Literal[True]:
  if not ver in CV_VERSIONS:
    print(f'FATAL: {ver} is not a supported Common Voice version.')
    import sys
    sys.exit(1)
  return True

def calc_dataset_prefix(ver) -> str | None:
  if is_version_valid(ver):
    inx: int = CV_VERSIONS.index(ver)
    return f'cv-corpus-{ver}-{CV_DATES[inx]}'

def get_locales (ver: str) -> list[str]:
  url: str = f"{CV_DATASET_BASE_URL}/{calc_dataset_prefix(ver)}.json"
  try:
    res: Any = urlopen(url)
  except:
    print(f'Metadata for version {ver} could not be located!')
    sys.exit()
  jdict: Any = json.loads(res.read())
  jlocales: Any = jdict["locales"]
  locales: list[str] = []
  for loc, data in jlocales.items():
    locales.append(loc)
  locales.sort()
  return locales
