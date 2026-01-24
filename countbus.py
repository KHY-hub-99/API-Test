import pandas as pd
import zipfile
import io

# 검사할 파일명
GTFS_FILE = "./data/south_korea_gtfs.zip"

def check_seoul_data():
    print(f"🔍 GTFS 데이터 내용 검사 중... ({GTFS_FILE})")
    
    try:
        with zipfile.ZipFile(GTFS_FILE) as z:
            # 1. 정류장(stops.txt) 확인
            with z.open("stops.txt") as f:
                stops = pd.read_csv(f)
                
                # 서울 공예박물관 근처(위도 37.57, 경도 126.98) 정류장이 있는지 확인
                seoul_stops = stops[
                    (stops['stop_lat'] > 37.57) & (stops['stop_lat'] < 37.58) &
                    (stops['stop_lon'] > 126.98) & (stops['stop_lon'] < 126.99)
                ]
                
                print(f"\n1️⃣ 서울 종로구 인근 정류장 개수: {len(seoul_stops)}개")
                if len(seoul_stops) > 0:
                    print(f"   👉 예시: {seoul_stops.iloc[0]['stop_name']} (ID: {seoul_stops.iloc[0]['stop_id']})")
                else:
                    print("   ❌ 경고: 이 파일에는 서울 도심 정류장 정보가 없습니다!")
                    return

            # 2. 2024년 5월 20일 운행 여부 확인
            with z.open("calendar.txt") as f:
                cal = pd.read_csv(f)
                # 월요일(monday)이 1이고, 날짜 범위에 20240520이 들어가는지
                target_date = 20240520
                active_services = cal[
                    (cal['monday'] == 1) & 
                    (cal['start_date'] <= target_date) & 
                    (cal['end_date'] >= target_date)
                ]
                print(f"\n2️⃣ 2024-05-20(월) 운행하는 서비스 ID 개수: {len(active_services)}개")
                if len(active_services) == 0:
                    print("   ❌ 경고: 해당 날짜에 운행하는 버스 스케줄이 없습니다.")
                else:
                    print("   👉 정상입니다. 운행 스케줄이 존재합니다.")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    check_seoul_data()