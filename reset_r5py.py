import os
import shutil
import zipfile
import glob

# ==========================================
# 설정: 본인의 파일 경로로 맞춰주세요
# ==========================================
DATA_DIR = "./data"
GTFS_FILE = "south_korea_gtfs.zip"  # 현재 사용 중인 GTFS 파일명
# ==========================================

def reset_network():
    print("🧹 [1/3] r5py 네트워크 캐시 삭제 중...")
    
    # network.dat 및 관련 캐시 파일 찾아서 삭제
    patterns = ["network.dat", "network.dat.mapdb*", "*.pbf.mapdb*"]
    deleted_count = 0
    
    for pattern in patterns:
        files = glob.glob(os.path.join(DATA_DIR, pattern))
        for f in files:
            try:
                if os.path.isdir(f): shutil.rmtree(f)
                else: os.remove(f)
                print(f"   🗑️ 삭제됨: {f}")
                deleted_count += 1
            except Exception as e:
                print(f"   ⚠️ 삭제 실패: {f} ({e})")
    
    if deleted_count == 0:
        print("   ✨ 삭제할 캐시 파일이 없습니다. (이미 깨끗함)")
    else:
        print("   ✅ 캐시 삭제 완료.")

def fix_gtfs_structure():
    print(f"\n🛠️ [2/3] GTFS 압축 구조 검사 및 수리: {GTFS_FILE}")
    gtfs_path = os.path.join(DATA_DIR, GTFS_FILE)
    
    if not os.path.exists(gtfs_path):
        print(f"   ❌ 오류: 파일을 찾을 수 없습니다 -> {gtfs_path}")
        return

    temp_dir = os.path.join(DATA_DIR, "temp_gtfs_fix_struct")
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    try:
        # 압축 해제
        with zipfile.ZipFile(gtfs_path, 'r') as z:
            z.extractall(temp_dir)
        
        # 내용물 확인
        files = os.listdir(temp_dir)
        has_txt = any(f.endswith(".txt") for f in files)
        
        # 폴더 안에 폴더가 있는 경우 (잘못된 구조)
        if not has_txt and len(files) == 1 and os.path.isdir(os.path.join(temp_dir, files[0])):
            nested_folder = os.path.join(temp_dir, files[0])
            print(f"   ⚠️ 잘못된 구조 발견! (폴더 안에 폴더가 있음: {files[0]})")
            print("   🔧 구조 평탄화(Flattening) 진행 중...")
            
            # 내부 파일들을 밖으로 꺼내고 다시 압축
            new_zip_path = os.path.join(DATA_DIR, "seoul_gtfs_repacked.zip")
            with zipfile.ZipFile(new_zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
                for root, _, filenames in os.walk(nested_folder):
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        z.write(file_path, filename) # 경로 떼고 파일명만 저장
            
            print(f"   🎉 수리 완료! 새 파일 생성됨: seoul_gtfs_repacked.zip")
            print("   👉 메인 코드에서 이 파일을 사용하세요!")
            return "seoul_gtfs_repacked.zip"
            
        else:
            print("   ✅ GTFS 구조가 정상입니다. (수리 불필요)")
            return GTFS_FILE

    except Exception as e:
        print(f"   ❌ 검사 중 오류: {e}")
        return GTFS_FILE
    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

def find_and_destroy_cache():
    print("🕵️‍♂️ 프로젝트 폴더 전체를 뒤져서 'network.dat'를 찾습니다...")
    
    # 현재 파이썬 파일이 있는 위치부터 시작
    root_dir = os.getcwd()
    found_count = 0
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # network.dat 또는 관련 파일 찾기
            if filename == "network.dat" or filename.startswith("network.dat."):
                full_path = os.path.join(dirpath, filename)
                try:
                    print(f"   🚩 발견! 삭제 중: {full_path}")
                    os.remove(full_path)
                    found_count += 1
                except Exception as e:
                    print(f"   ⚠️ 삭제 실패 (사용 중일 수 있음): {full_path} / {e}")
    
    if found_count == 0:
        print("   🤷‍♂️ 진짜로 파일이 없습니다. (이미 지워졌거나, 생성된 적이 없음)")
    else:
        print(f"   ✅ 총 {found_count}개의 캐시 파일을 삭제했습니다.")

if __name__ == "__main__":
    reset_network()
    new_gtfs = fix_gtfs_structure()
    
    print("\n🚀 [3/3] 준비 완료!")
    if new_gtfs != GTFS_FILE:
        print(f"⚠️ 중요: 메인 코드(test.py)에서 GTFS 파일명을 '{new_gtfs}'로 바꿔주세요!")
    else:
        print("✅ 메인 코드를 바로 다시 실행하시면 됩니다. (초기 로딩 시간 걸림)")
        
    find_and_destroy_cache()