import pandas as pd
import zipfile
import pandas as pd
from io import BytesIO

gtfs_zip_path = "./data/seoul_gtfs.zip"

# ZIP 파일 열기
with zipfile.ZipFile(gtfs_zip_path, 'r') as zip_ref:
    # stops.txt 불러오기
    with zip_ref.open('stops.txt') as f:
        stops = pd.read_csv(f, dtype=str)
    
    # transfers.txt 불러오기
    with zip_ref.open('transfers.txt') as f:
        transfers = pd.read_csv(f, dtype=str)

# 샘플 확인
print("stops.txt 샘플:")
print(stops.head())

print("\ntransfers.txt 샘플:")
print(transfers.head())

# stops.txt의 stop_id 집합
stop_ids = set(stops['stop_id'])

# 존재하지 않는 to_stop_id와 from_stop_id 찾기
invalid_to = transfers[~transfers['to_stop_id'].isin(stop_ids)]
invalid_from = transfers[~transfers['from_stop_id'].isin(stop_ids)]

print(f"존재하지 않는 to_stop_id 개수: {len(invalid_to)}")
print(f"존재하지 않는 from_stop_id 개수: {len(invalid_from)}")

# 문제 있는 행들을 별도 CSV로 저장 (검토용)
invalid_to.to_csv("invalid_to_stop_id.csv", index=False)
invalid_from.to_csv("invalid_from_stop_id.csv", index=False)

# 존재하지 않는 stop_id 제거한 transfers 파일 새로 저장
clean_transfers = transfers[
    transfers['to_stop_id'].isin(stop_ids) &
    transfers['from_stop_id'].isin(stop_ids)
]
clean_transfers.to_csv("transfers_cleaned.txt", index=False)

print("정상적인 transfers 파일 'transfers_cleaned.txt'로 저장 완료!")

transfers_edit = pd.read_csv("transfers_cleaned.txt")
print(len(transfers_edit))