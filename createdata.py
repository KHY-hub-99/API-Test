import random
import pandas as pd

areas = ["종로구", "강남구", "마포구", "성수동", "홍대", "이태원", "잠실"]
categories = ["관광지", "카페", "식당", "박물관", "공원", "시장", "숙박"]
mood_pool = ["감성", "조용한", "데이트", "힐링", "혼자", "활동적", "야경", "가족"]

stay_time_map = {
    "관광지": 90,
    "카페": 50,
    "식당": 70,
    "박물관": 120,
    "공원": 60,
    "시장": 80,
    "숙박": 0
}

def random_place_name(category, area):
    suffix = {
        "관광지": ["명소", "전망대", "거리", "마을"],
        "카페": ["감성카페", "뷰카페", "로스터리", "브런치"],
        "식당": ["맛집", "식당", "국밥집", "파스타집"],
        "박물관": ["박물관", "전시관"],
        "공원": ["공원", "산책로"],
        "시장": ["시장", "야시장"],
        "숙박": ["호텔", "게스트하우스", "모텔"]
    }
    return f"{area} {random.choice(suffix[category])}"

rows = []

for i in range(1, 3001):
    category = random.choice(categories)
    area = random.choice(areas)
    moods = random.sample(mood_pool, k=random.randint(2, 4))

    row = {
        "place_id": f"P{i:04d}",
        "name": random_place_name(category, area),
        "category": category,
        "lat": round(37.4 + random.random() * 0.3, 6),
        "lng": round(126.8 + random.random() * 0.4, 6),
        "avg_stay_min": stay_time_map[category],
        "mood_tags": ", ".join(moods),
        "crowd_level": round(random.uniform(0.1, 0.9), 2),
        "area": area
    }

    rows.append(row)

df = pd.DataFrame(rows)

df.to_excel("places_3000.xlsx", index=False)