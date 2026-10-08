"""Listuje pliki w zdalnym zipie HaGRIDv2 512px bez pobierania go (zapytania HTTP Range)."""
import truststore
truststore.inject_into_ssl()

import pandas as pd
from remotezip import RemoteZip

URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/hagridv2_512.zip"

with RemoteZip(URL) as z:
    rows = [(i.filename, i.file_size, i.compress_size) for i in z.infolist()]
df = pd.DataFrame(rows, columns=["path", "size", "compress_size"])
df.to_parquet("data/raw/remote_512_index.parquet")
print(len(df))
print(df.head(10).to_string())
