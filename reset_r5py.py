import gc

# 1. 기존 네트워크 변수 삭제
if 'transport_network' in globals():
    del transport_network
    print("🗑️ transport_network 변수 삭제됨")

if "tn" in globals():
    del tn
    print("🗑️ tn 변수 삭제됨")

# 2. 가비지 컬렉터 강제 실행 (메모리 청소)
gc.collect()
print("✨ 메모리 청소 완료")