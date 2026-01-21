# # 거리 계산
# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371  # km
#     phi1 = math.radians(lat1)
#     phi2 = math.radians(lat2)
#     dphi = math.radians(lat2 - lat1)
#     dlambda = math.radians(lon2 - lon1)

#     a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
#     return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

# # 주변 음식점 찾기
# def find_near_restaurants(place, restaurants, k=2):
#     lat, lng = place["lat"], place["lng"]
#     dists = []

#     for r in restaurants:
#         d = haversine(lat, lng, r["lat"], r["lng"])
#         dists.append((d, r))

#     dists.sort(key=lambda x: x[0])
#     return [r for _, r in dists[:k]]

# # 반드시 plans 구조로 접근
# if "plans" not in result:
#     raise ValueError("Gemini JSON에 plans 필드가 없습니다.")

# plans = result["plans"]

# final_result = {}

# for day_key, day_places in plans.items():

#     # 방어 코드
#     if not isinstance(day_places, list):
#         print(f"⚠ {day_key} 구조가 리스트가 아닙니다. 건너뜁니다.")
#         continue

#     day_restaurants = []

#     for place in day_places:
#         if "lat" not in place or "lng" not in place:
#             continue

#         near = find_near_restaurants(place, restaurants)
#         day_restaurants.extend(near)

#     # 중복 제거 + 하루 2곳
#     unique = {r["name"]: r for r in day_restaurants}.values()
#     day_restaurants = list(unique)[:2]

#     final_result[day_key] = {
#         "route": day_places,
#         "restaurants": day_restaurants
#     }

# print("\n====== 최종 결과 ======\n")
# print(json.dumps(final_result, ensure_ascii=False, indent=2))