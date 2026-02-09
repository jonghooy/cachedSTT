# NeMo Cache-Aware Streaming 한국어 ASR 파일럿 가이드

> Timbel 1000시간 파일럿 훈련을 위한 완전 가이드

## 📋 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [환경 설정](#2-환경-설정)
3. [데이터 준비](#3-데이터-준비)
4. [토크나이저 생성](#4-토크나이저-생성)
5. [훈련 설정](#5-훈련-설정)
6. [훈련 실행](#6-훈련-실행)
7. [평가 및 테스트](#7-평가-및-테스트)
8. [스트리밍 추론 테스트](#8-스트리밍-추론-테스트)
9. [성능 분석](#9-성능-분석)
10. [Go/No-Go 결정](#10-gono-go-결정)
11. [트러블슈팅](#11-트러블슈팅)
12. [스케일업 계획](#12-스케일업-계획)

---

## 1. 프로젝트 개요

### 1.1 목표

- **1000시간** 한국어 데이터로 Cache-Aware Streaming ASR 파일럿 훈련
- 기술 검증 및 성능 기준선 확립
- 38,000시간 풀 스케일 훈련 전 리스크 최소화

### 1.2 데이터 구성

| 소스 | 샘플레이트 | 시간 | 비율 |
|------|-----------|------|------|
| 일반 음성 | 16KHz | 800시간 | 80% |
| 전화 통화 | 8KHz → 16KHz | 200시간 | 20% |
| **합계** | 16KHz | **1000시간** | 100% |

### 1.3 예상 리소스

| 항목 | 사양 |
|------|------|
| GPU | 1-2x A100 (40GB) 또는 1x RTX 4090 |
| RAM | 64GB+ |
| Storage | 500GB+ (오디오 + 체크포인트) |
| 훈련 시간 | 2-4일 |
| 예상 비용 | $300-500 (클라우드) |

### 1.4 예상 성능

| 메트릭 | 목표 |
|--------|------|
| 일반 음성 CER | < 10% |
| 전화 음성 CER | < 15% |
| 스트리밍 지연 | 80-1040ms (설정 가능) |

---

## 2. 환경 설정

### 2.1 시스템 요구사항

```bash
# OS: Ubuntu 20.04/22.04
# CUDA: 11.8 이상
# Python: 3.10+

# CUDA 버전 확인
nvidia-smi
nvcc --version
```

### 2.2 프로젝트 디렉토리 구조

```bash
mkdir -p ~/timbel-asr-pilot
cd ~/timbel-asr-pilot

# 디렉토리 구조 생성
mkdir -p {data,configs,scripts,tokenizer,experiments,logs,outputs}
mkdir -p data/{raw_16k,raw_8k,processed,manifests}

# 구조 확인
tree -L 2
# timbel-asr-pilot/
# ├── configs/          # 훈련 설정 파일
# ├── data/
# │   ├── raw_16k/      # 원본 16KHz 오디오
# │   ├── raw_8k/       # 원본 8KHz 오디오
# │   ├── processed/    # 처리된 오디오 (모두 16KHz)
# │   └── manifests/    # JSON manifest 파일
# ├── experiments/      # 훈련 체크포인트
# ├── logs/             # 훈련 로그
# ├── outputs/          # 추론 결과
# ├── scripts/          # 유틸리티 스크립트
# └── tokenizer/        # 토크나이저 파일
```

### 2.3 NeMo 설치

```bash
# 가상환경 생성
conda create -n nemo-asr python=3.10 -y
conda activate nemo-asr

# PyTorch 설치 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cython 먼저 설치
pip install Cython

# NeMo 설치 (최신 main 브랜치)
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[all]

# 추가 의존성
pip install librosa soundfile pandas matplotlib seaborn tqdm
pip install tensorboard wandb  # 모니터링용
pip install jiwer  # WER 계산용

# 설치 확인
python -c "import nemo; print(f'NeMo version: {nemo.__version__}')"
python -c "import nemo.collections.asr as nemo_asr; print('ASR module loaded successfully')"
```

### 2.4 NeMo 레포지토리 클론 (스크립트용)

```bash
cd ~/timbel-asr-pilot
git clone https://github.com/NVIDIA/NeMo.git
export NEMO_ROOT=~/timbel-asr-pilot/NeMo
```

---

## 3. 데이터 준비

### 3.1 Manifest 포맷 이해

NeMo는 JSON Lines (.json) 형식의 manifest 파일을 사용합니다:

```json
{"audio_filepath": "/path/to/audio.wav", "text": "transcription text", "duration": 3.5}
{"audio_filepath": "/path/to/audio2.wav", "text": "another transcription", "duration": 2.1}
```

필수 필드:
- `audio_filepath`: 오디오 파일 경로 (절대 경로 권장)
- `text`: 정규화된 텍스트 전사
- `duration`: 오디오 길이 (초)

### 3.2 데이터 샘플링 스크립트

**`scripts/sample_dataset.py`** 생성:

```python
#!/usr/bin/env python3
"""
전체 데이터에서 파일럿용 1000시간 샘플링
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_manifest(filepath):
    """Manifest 로드"""
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def save_manifest(entries, filepath):
    """Manifest 저장"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Saved {len(entries)} entries to {filepath}")


def calculate_total_duration(entries):
    """총 duration 계산 (시간 단위)"""
    total_seconds = sum(e.get('duration', 0) for e in entries)
    return total_seconds / 3600


def sample_by_duration(entries, target_hours, seed=42):
    """
    목표 시간에 맞게 랜덤 샘플링
    """
    random.seed(seed)
    random.shuffle(entries)
    
    sampled = []
    current_hours = 0
    
    for entry in entries:
        duration_hours = entry.get('duration', 0) / 3600
        if current_hours + duration_hours <= target_hours:
            sampled.append(entry)
            current_hours += duration_hours
        
        if current_hours >= target_hours:
            break
    
    return sampled


def sample_dataset(
    manifest_16k: str,
    manifest_8k: str,
    output_dir: str,
    total_hours: int = 1000,
    ratio_16k: float = 0.8,
    seed: int = 42
):
    """
    16KHz와 8KHz 데이터를 비율에 맞게 샘플링
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 목표 시간 계산
    hours_16k = int(total_hours * ratio_16k)
    hours_8k = total_hours - hours_16k
    
    print(f"Target: {hours_16k}h (16KHz) + {hours_8k}h (8KHz) = {total_hours}h total")
    
    # 16KHz 데이터 샘플링
    print(f"\n[1/2] Loading 16KHz manifest: {manifest_16k}")
    entries_16k = load_manifest(manifest_16k)
    print(f"  Total entries: {len(entries_16k)}")
    print(f"  Total duration: {calculate_total_duration(entries_16k):.1f}h")
    
    sampled_16k = sample_by_duration(entries_16k, hours_16k, seed)
    for entry in sampled_16k:
        entry['domain'] = 'wideband'
        entry['orig_sr'] = 16000
    print(f"  Sampled: {len(sampled_16k)} entries ({calculate_total_duration(sampled_16k):.1f}h)")
    
    # 8KHz 데이터 샘플링
    print(f"\n[2/2] Loading 8KHz manifest: {manifest_8k}")
    entries_8k = load_manifest(manifest_8k)
    print(f"  Total entries: {len(entries_8k)}")
    print(f"  Total duration: {calculate_total_duration(entries_8k):.1f}h")
    
    sampled_8k = sample_by_duration(entries_8k, hours_8k, seed + 1)
    for entry in sampled_8k:
        entry['domain'] = 'telephony'
        entry['orig_sr'] = 8000
    print(f"  Sampled: {len(sampled_8k)} entries ({calculate_total_duration(sampled_8k):.1f}h)")
    
    # 통합 및 셔플
    combined = sampled_16k + sampled_8k
    random.seed(seed + 2)
    random.shuffle(combined)
    
    # Train/Val/Test 분할 (90/5/5)
    n = len(combined)
    n_train = int(n * 0.90)
    n_val = int(n * 0.05)
    
    train_set = combined[:n_train]
    val_set = combined[n_train:n_train + n_val]
    test_set = combined[n_train + n_val:]
    
    # 저장
    save_manifest(train_set, output_dir / 'train_manifest.json')
    save_manifest(val_set, output_dir / 'val_manifest.json')
    save_manifest(test_set, output_dir / 'test_manifest.json')
    save_manifest(combined, output_dir / 'all_manifest.json')
    
    # 통계 출력
    print(f"\n=== Dataset Statistics ===")
    print(f"Train: {len(train_set)} entries ({calculate_total_duration(train_set):.1f}h)")
    print(f"Val:   {len(val_set)} entries ({calculate_total_duration(val_set):.1f}h)")
    print(f"Test:  {len(test_set)} entries ({calculate_total_duration(test_set):.1f}h)")
    print(f"Total: {len(combined)} entries ({calculate_total_duration(combined):.1f}h)")
    
    # 도메인별 통계
    domain_stats = defaultdict(lambda: {'count': 0, 'duration': 0})
    for entry in combined:
        domain = entry.get('domain', 'unknown')
        domain_stats[domain]['count'] += 1
        domain_stats[domain]['duration'] += entry.get('duration', 0)
    
    print(f"\n=== Domain Distribution ===")
    for domain, stats in domain_stats.items():
        hours = stats['duration'] / 3600
        print(f"{domain}: {stats['count']} entries ({hours:.1f}h)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample dataset for pilot training')
    parser.add_argument('--manifest_16k', required=True, help='Path to 16KHz manifest')
    parser.add_argument('--manifest_8k', required=True, help='Path to 8KHz manifest')
    parser.add_argument('--output_dir', default='./data/manifests', help='Output directory')
    parser.add_argument('--total_hours', type=int, default=1000, help='Total hours to sample')
    parser.add_argument('--ratio_16k', type=float, default=0.8, help='Ratio of 16KHz data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    sample_dataset(
        manifest_16k=args.manifest_16k,
        manifest_8k=args.manifest_8k,
        output_dir=args.output_dir,
        total_hours=args.total_hours,
        ratio_16k=args.ratio_16k,
        seed=args.seed
    )
```

### 3.3 8KHz → 16KHz 리샘플링 스크립트

**`scripts/resample_audio.py`** 생성:

```python
#!/usr/bin/env python3
"""
8KHz 오디오를 16KHz로 리샘플링
"""

import json
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import os


def resample_single_file(args):
    """단일 파일 리샘플링"""
    input_path, output_path, target_sr = args
    
    try:
        # 오디오 로드 (원본 샘플레이트로)
        audio, sr = librosa.load(input_path, sr=None)
        
        # 리샘플링 필요 시
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 저장
        sf.write(output_path, audio, target_sr)
        
        return {
            'input': input_path,
            'output': output_path,
            'success': True,
            'orig_sr': sr,
            'target_sr': target_sr
        }
    except Exception as e:
        return {
            'input': input_path,
            'output': output_path,
            'success': False,
            'error': str(e)
        }


def resample_from_manifest(
    input_manifest: str,
    output_dir: str,
    output_manifest: str,
    target_sr: int = 16000,
    num_workers: int = 16
):
    """
    Manifest 기반으로 오디오 리샘플링
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Manifest 로드
    entries = []
    with open(input_manifest, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    print(f"Loaded {len(entries)} entries from {input_manifest}")
    
    # 작업 목록 생성
    tasks = []
    for entry in entries:
        input_path = entry['audio_filepath']
        filename = Path(input_path).name
        output_path = str(output_dir / filename)
        tasks.append((input_path, output_path, target_sr))
    
    # 병렬 처리
    results = []
    failed = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(resample_single_file, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Resampling"):
            result = future.result()
            if result['success']:
                results.append(result)
            else:
                failed.append(result)
    
    print(f"\nCompleted: {len(results)} files")
    print(f"Failed: {len(failed)} files")
    
    if failed:
        print("\nFailed files:")
        for f in failed[:10]:  # 처음 10개만 출력
            print(f"  {f['input']}: {f['error']}")
    
    # 새 manifest 생성
    new_entries = []
    output_path_map = {r['input']: r['output'] for r in results}
    
    for entry in entries:
        if entry['audio_filepath'] in output_path_map:
            new_entry = entry.copy()
            new_entry['audio_filepath'] = output_path_map[entry['audio_filepath']]
            new_entry['orig_sr'] = 8000  # 원본이 8KHz였음을 표시
            new_entries.append(new_entry)
    
    # 새 manifest 저장
    with open(output_manifest, 'w', encoding='utf-8') as f:
        for entry in new_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\nSaved new manifest: {output_manifest}")
    print(f"Total entries: {len(new_entries)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resample audio files')
    parser.add_argument('--input_manifest', required=True, help='Input manifest path')
    parser.add_argument('--output_dir', required=True, help='Output directory for resampled audio')
    parser.add_argument('--output_manifest', required=True, help='Output manifest path')
    parser.add_argument('--target_sr', type=int, default=16000, help='Target sample rate')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    resample_from_manifest(
        input_manifest=args.input_manifest,
        output_dir=args.output_dir,
        output_manifest=args.output_manifest,
        target_sr=args.target_sr,
        num_workers=args.num_workers
    )
```

### 3.4 텍스트 정규화 스크립트

**`scripts/normalize_text.py`** 생성:

```python
#!/usr/bin/env python3
"""
한국어 텍스트 정규화
- 숫자 → 한글
- 특수문자 처리
- 공백 정규화
"""

import re
import json
from pathlib import Path
import argparse
from tqdm import tqdm


# 숫자 → 한글 변환
DIGIT_TO_KOREAN = {
    '0': '영', '1': '일', '2': '이', '3': '삼', '4': '사',
    '5': '오', '6': '육', '7': '칠', '8': '팔', '9': '구'
}

# 단위 숫자
UNITS = ['', '십', '백', '천']
LARGE_UNITS = ['', '만', '억', '조']


def number_to_korean(num_str):
    """숫자를 한글로 변환 (간단 버전)"""
    try:
        num = int(num_str)
        if num == 0:
            return '영'
        
        result = ''
        num_str = str(num)
        length = len(num_str)
        
        for i, digit in enumerate(num_str):
            if digit != '0':
                position = length - i - 1
                unit_idx = position % 4
                large_unit_idx = position // 4
                
                if digit == '1' and unit_idx > 0:
                    result += UNITS[unit_idx]
                else:
                    result += DIGIT_TO_KOREAN[digit] + UNITS[unit_idx]
                
                if unit_idx == 0 and large_unit_idx > 0:
                    result += LARGE_UNITS[large_unit_idx]
        
        return result
    except:
        return num_str


def normalize_korean_text(text):
    """한국어 텍스트 정규화"""
    
    # 1. 기본 정리
    text = text.strip()
    
    # 2. 숫자 → 한글 변환 (선택적 - 필요시 활성화)
    # text = re.sub(r'\d+', lambda m: number_to_korean(m.group()), text)
    
    # 3. 허용 문자만 유지 (한글, 영문, 숫자, 공백, 기본 문장부호)
    # 한글: 가-힣, ㄱ-ㅎ, ㅏ-ㅣ
    text = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ.,!?\'"-]', ' ', text)
    
    # 4. 연속 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 5. 앞뒤 공백 제거
    text = text.strip()
    
    # 6. 소문자 변환 (영문 포함 시)
    text = text.lower()
    
    return text


def normalize_manifest(input_path, output_path):
    """Manifest 내 모든 텍스트 정규화"""
    
    entries = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    normalized = []
    empty_count = 0
    
    for entry in tqdm(entries, desc="Normalizing"):
        new_entry = entry.copy()
        original_text = entry.get('text', '')
        normalized_text = normalize_korean_text(original_text)
        
        if not normalized_text:
            empty_count += 1
            continue
        
        new_entry['text'] = normalized_text
        normalized.append(new_entry)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in normalized:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Input: {len(entries)} entries")
    print(f"Output: {len(normalized)} entries")
    print(f"Removed (empty text): {empty_count} entries")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Normalize Korean text in manifest')
    parser.add_argument('--input', required=True, help='Input manifest path')
    parser.add_argument('--output', required=True, help='Output manifest path')
    
    args = parser.parse_args()
    normalize_manifest(args.input, args.output)
```

### 3.5 데이터 검증 스크립트

**`scripts/validate_data.py`** 생성:

```python
#!/usr/bin/env python3
"""
데이터 검증 스크립트
- 오디오 파일 존재 확인
- Duration 검증
- 텍스트 검증
"""

import json
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import librosa
from collections import defaultdict


def validate_manifest(manifest_path, check_audio=True, fix_duration=False):
    """Manifest 검증"""
    
    entries = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    print(f"Validating {len(entries)} entries from {manifest_path}")
    
    issues = defaultdict(list)
    valid_entries = []
    
    for entry in tqdm(entries, desc="Validating"):
        audio_path = entry.get('audio_filepath', '')
        text = entry.get('text', '')
        duration = entry.get('duration', 0)
        
        has_issue = False
        
        # 1. 오디오 파일 존재 확인
        if check_audio and not os.path.exists(audio_path):
            issues['missing_audio'].append(audio_path)
            has_issue = True
            continue
        
        # 2. 텍스트 확인
        if not text or len(text.strip()) == 0:
            issues['empty_text'].append(audio_path)
            has_issue = True
            continue
        
        # 3. Duration 확인
        if duration <= 0:
            issues['invalid_duration'].append(audio_path)
            
            if fix_duration and os.path.exists(audio_path):
                try:
                    audio, sr = librosa.load(audio_path, sr=None)
                    entry['duration'] = len(audio) / sr
                except:
                    has_issue = True
                    continue
            else:
                has_issue = True
                continue
        
        # 4. Duration 범위 확인 (0.5초 ~ 30초)
        if duration < 0.5:
            issues['too_short'].append((audio_path, duration))
            has_issue = True
            continue
        
        if duration > 30:
            issues['too_long'].append((audio_path, duration))
            # 긴 파일은 제외하지 않고 경고만 (선택적)
        
        if not has_issue:
            valid_entries.append(entry)
    
    # 결과 출력
    print(f"\n=== Validation Results ===")
    print(f"Total entries: {len(entries)}")
    print(f"Valid entries: {len(valid_entries)}")
    print(f"Invalid entries: {len(entries) - len(valid_entries)}")
    
    print(f"\n=== Issues ===")
    for issue_type, items in issues.items():
        print(f"{issue_type}: {len(items)}")
        if len(items) > 0 and len(items) <= 5:
            for item in items:
                print(f"  - {item}")
    
    return valid_entries, issues


def save_validated_manifest(entries, output_path):
    """검증된 manifest 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"\nSaved validated manifest: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate manifest')
    parser.add_argument('--input', required=True, help='Input manifest path')
    parser.add_argument('--output', help='Output manifest path (validated)')
    parser.add_argument('--check_audio', action='store_true', help='Check if audio files exist')
    parser.add_argument('--fix_duration', action='store_true', help='Fix duration by loading audio')
    
    args = parser.parse_args()
    
    valid_entries, issues = validate_manifest(
        args.input,
        check_audio=args.check_audio,
        fix_duration=args.fix_duration
    )
    
    if args.output:
        save_validated_manifest(valid_entries, args.output)
```

### 3.6 데이터 준비 실행

```bash
cd ~/timbel-asr-pilot

# 1. 8KHz 데이터 리샘플링 (8KHz → 16KHz)
python scripts/resample_audio.py \
    --input_manifest /path/to/original_8k_manifest.json \
    --output_dir ./data/processed/phone_16k \
    --output_manifest ./data/manifests/phone_16k_resampled.json \
    --target_sr 16000 \
    --num_workers 32

# 2. 텍스트 정규화 (16KHz 데이터)
python scripts/normalize_text.py \
    --input /path/to/original_16k_manifest.json \
    --output ./data/manifests/general_16k_normalized.json

# 3. 텍스트 정규화 (리샘플링된 8KHz 데이터)
python scripts/normalize_text.py \
    --input ./data/manifests/phone_16k_resampled.json \
    --output ./data/manifests/phone_16k_normalized.json

# 4. 1000시간 샘플링
python scripts/sample_dataset.py \
    --manifest_16k ./data/manifests/general_16k_normalized.json \
    --manifest_8k ./data/manifests/phone_16k_normalized.json \
    --output_dir ./data/manifests \
    --total_hours 1000 \
    --ratio_16k 0.8 \
    --seed 42

# 5. 데이터 검증
python scripts/validate_data.py \
    --input ./data/manifests/train_manifest.json \
    --output ./data/manifests/train_manifest_validated.json \
    --check_audio

python scripts/validate_data.py \
    --input ./data/manifests/val_manifest.json \
    --output ./data/manifests/val_manifest_validated.json \
    --check_audio

python scripts/validate_data.py \
    --input ./data/manifests/test_manifest.json \
    --output ./data/manifests/test_manifest_validated.json \
    --check_audio
```

---

## 4. 토크나이저 생성

### 4.1 SentencePiece 토크나이저 생성

```bash
# NeMo 스크립트 사용
python $NEMO_ROOT/scripts/tokenizers/process_asr_text_tokenizer.py \
    --manifest=./data/manifests/train_manifest_validated.json \
    --data_root=./tokenizer \
    --vocab_size=1024 \
    --tokenizer=spe \
    --spe_type=bpe \
    --spe_character_coverage=1.0 \
    --spe_sample_size=10000000

# 결과 확인
ls -la ./tokenizer/
# tokenizer_spe_bpe_v1024/
# ├── tokenizer.model
# └── vocab.txt
```

### 4.2 토크나이저 테스트

```python
# scripts/test_tokenizer.py
import sentencepiece as spm

# 토크나이저 로드
sp = spm.SentencePieceProcessor()
sp.load('./tokenizer/tokenizer_spe_bpe_v1024/tokenizer.model')

# 테스트
test_texts = [
    "안녕하세요 반갑습니다",
    "오늘 회의 시작하겠습니다",
    "전화 연결 부탁드립니다",
    "음성 인식 테스트입니다"
]

for text in test_texts:
    tokens = sp.encode_as_pieces(text)
    ids = sp.encode_as_ids(text)
    decoded = sp.decode(ids)
    
    print(f"Original: {text}")
    print(f"Tokens:   {tokens}")
    print(f"IDs:      {ids}")
    print(f"Decoded:  {decoded}")
    print()

print(f"Vocab size: {sp.get_piece_size()}")
```

---

## 5. 훈련 설정

### 5.1 Config 파일 생성

**`configs/fastconformer_hybrid_streaming_ko_pilot.yaml`** 생성:

```yaml
# Timbel Korean ASR Pilot - Cache-Aware Streaming FastConformer
# 1000시간 파일럿 훈련용

name: "Timbel-Korean-FastConformer-Streaming-Pilot"

model:
  sample_rate: 16000
  
  # 로그 설정
  log_prediction: true
  
  # ===== 토크나이저 =====
  tokenizer:
    dir: ./tokenizer/tokenizer_spe_bpe_v1024
    type: bpe
  
  # ===== Preprocessor (Feature Extraction) =====
  preprocessor:
    _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
    sample_rate: 16000
    normalize: "per_feature"
    window_size: 0.025
    window_stride: 0.01
    window: "hann"
    features: 80
    n_fft: 512
    frame_splicing: 1
    dither: 0.00001
    pad_to: 0
    stft_conv: false
  
  # ===== Spec Augmentation =====
  spec_augment:
    _target_: nemo.collections.asr.modules.SpectrogramAugmentation
    freq_masks: 2
    freq_width: 27
    time_masks: 10
    time_width: 0.05
  
  # ===== Encoder (FastConformer - Cache-Aware) =====
  encoder:
    _target_: nemo.collections.asr.modules.ConformerEncoder
    feat_in: 80
    feat_out: -1
    n_layers: 17                    # Medium-Large (빠른 실험용)
    d_model: 512
    
    # Subsampling (FastConformer 8x)
    subsampling: dw_striding
    subsampling_factor: 8
    subsampling_conv_channels: 256
    causal_downsampling: false
    
    # ⭐ Cache-Aware Streaming 핵심 설정
    att_context_style: "chunked_limited"
    att_context_size: [70, 13]      # [left_context, right_context] in frames
    # 지원 지연: 80ms × right_context = 1040ms (기본)
    # Multi-latency 훈련 시: [[70,0], [70,1], [70,13], [70,33]]
    
    # Self-Attention
    self_attention_model: rel_pos
    n_heads: 8
    xscaling: true
    untie_biases: true
    pos_emb_max_len: 5000
    
    # Feed-Forward
    ff_expansion_factor: 4
    
    # Convolution
    conv_kernel_size: 9
    conv_norm_type: batch_norm
    conv_context_size: null
    
    # Dropout
    dropout: 0.1
    dropout_pre_encoder: 0.1
    dropout_emb: 0.0
    dropout_att: 0.1
  
  # ===== Transducer Decoder =====
  decoder:
    _target_: nemo.collections.asr.modules.RNNTDecoder
    normalization_mode: null
    random_state_sampling: false
    blank_as_pad: true
    
    prednet:
      pred_hidden: 640
      pred_rnn_layers: 1
      t_max: null
      dropout: 0.2
  
  # ===== Joint Network =====
  joint:
    _target_: nemo.collections.asr.modules.RNNTJoint
    log_softmax: null
    preserve_memory: false
    fuse_loss_wer: true
    fused_batch_size: 16
    
    jointnet:
      joint_hidden: 640
      activation: relu
      dropout: 0.2
      num_extra_outputs: 0
  
  # ===== Decoding =====
  decoding:
    strategy: greedy_batch
    greedy:
      max_symbols: 10
  
  # ===== Loss =====
  loss:
    loss_name: default
    warprnnt_numba_kwargs:
      fastemit_lambda: 0.001   # 스트리밍 지연 감소
      clamp: -1.0
  
  # ===== CTC Auxiliary Loss (Hybrid) =====
  aux_ctc:
    ctc_loss_weight: 0.3        # CTC 30%, Transducer 70%
    use_cer: true
    ctc_reduction: mean_batch
    decoder:
      _target_: nemo.collections.asr.modules.ConvASRDecoder
      feat_in: null
      num_classes: -1
      vocabulary: []
    decoding:
      strategy: greedy
  
  # ===== Optimizer =====
  optim:
    name: adamw
    lr: 0.001
    betas: [0.9, 0.98]
    weight_decay: 0.001
    
    sched:
      name: NoamAnnealing
      d_model: 512
      warmup_steps: 5000
      warmup_ratio: null
      min_lr: 1.0e-6

# ===== Trainer =====
trainer:
  devices: 1                      # GPU 수 (파일럿은 1-2개)
  num_nodes: 1
  accelerator: gpu
  strategy: auto                  # 단일 GPU면 auto, 멀티면 ddp
  
  precision: 16-mixed             # Mixed precision (속도 향상)
  
  max_epochs: 150                 # 파일럿: 150 에폭
  max_steps: -1
  
  accumulate_grad_batches: 4      # Effective batch = batch_size × 4
  gradient_clip_val: 1.0
  
  enable_checkpointing: true
  logger: true
  log_every_n_steps: 50
  
  val_check_interval: 0.5         # 에폭당 2번 validation
  check_val_every_n_epoch: 1
  num_sanity_val_steps: 0
  
  benchmark: false
  enable_model_summary: true

# ===== Experiment Manager =====
exp_manager:
  exp_dir: ./experiments
  name: ${name}
  
  create_tensorboard_logger: true
  create_wandb_logger: false      # W&B 사용 시 true
  wandb_logger_kwargs:
    name: ${name}
    project: timbel-asr-pilot
  
  create_checkpoint_callback: true
  checkpoint_callback_params:
    monitor: val_wer
    mode: min
    save_top_k: 5
    every_n_epochs: 1
    always_save_nemo: true
  
  resume_if_exists: true
  resume_ignore_no_checkpoint: true
```

### 5.2 데이터 로더 설정 추가

Config 파일에 데이터 로더 설정 추가:

```yaml
# configs/fastconformer_hybrid_streaming_ko_pilot.yaml 에 추가

model:
  # ... (위의 설정들)
  
  # ===== Train Dataset =====
  train_ds:
    manifest_filepath: ./data/manifests/train_manifest_validated.json
    sample_rate: 16000
    batch_size: 16                # GPU 메모리에 따라 조정
    shuffle: true
    num_workers: 8
    pin_memory: true
    
    trim_silence: false
    max_duration: 20.0            # 20초 이상 샘플 제외
    min_duration: 0.5             # 0.5초 미만 제외
    
    # Bucketing (길이별 그룹핑)
    bucketing_strategy: synced_randomized
    bucketing_batch_size: null
  
  # ===== Validation Dataset =====
  validation_ds:
    manifest_filepath: ./data/manifests/val_manifest_validated.json
    sample_rate: 16000
    batch_size: 16
    shuffle: false
    num_workers: 4
    pin_memory: true
    
    max_duration: 20.0
    min_duration: 0.5
  
  # ===== Test Dataset =====
  test_ds:
    manifest_filepath: ./data/manifests/test_manifest_validated.json
    sample_rate: 16000
    batch_size: 16
    shuffle: false
    num_workers: 4
    pin_memory: true
```

---

## 6. 훈련 실행

### 6.1 단일 GPU 훈련

```bash
cd ~/timbel-asr-pilot

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0
export NEMO_ROOT=~/timbel-asr-pilot/NeMo

# 훈련 실행
python $NEMO_ROOT/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
    --config-path=../../../configs \
    --config-name=fastconformer_hybrid_streaming_ko_pilot \
    trainer.devices=1 \
    trainer.max_epochs=150 \
    model.train_ds.batch_size=16 \
    model.validation_ds.batch_size=16 \
    exp_manager.name=pilot_1k_v1
```

### 6.2 Multi-GPU 훈련 (2-4 GPU)

```bash
# 2 GPU
python $NEMO_ROOT/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
    --config-path=../../../configs \
    --config-name=fastconformer_hybrid_streaming_ko_pilot \
    trainer.devices=2 \
    trainer.strategy=ddp \
    model.train_ds.batch_size=16 \
    +model.train_ds.num_workers=8 \
    exp_manager.name=pilot_1k_2gpu

# 4 GPU
python $NEMO_ROOT/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
    --config-path=../../../configs \
    --config-name=fastconformer_hybrid_streaming_ko_pilot \
    trainer.devices=4 \
    trainer.strategy=ddp \
    model.train_ds.batch_size=12 \
    exp_manager.name=pilot_1k_4gpu
```

### 6.3 훈련 모니터링

```bash
# TensorBoard
tensorboard --logdir=./experiments --port=6006

# 실시간 로그 확인
tail -f ./experiments/Timbel-Korean-FastConformer-Streaming-Pilot/*/nemo_log_globalrank-0_localrank-0.txt
```

### 6.4 훈련 재개 (중단 시)

```bash
# 체크포인트에서 자동 재개 (resume_if_exists: true 설정 시)
python $NEMO_ROOT/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
    --config-path=../../../configs \
    --config-name=fastconformer_hybrid_streaming_ko_pilot \
    trainer.devices=1 \
    exp_manager.name=pilot_1k_v1 \
    exp_manager.resume_if_exists=true
```

---

## 7. 평가 및 테스트

### 7.1 배치 모드 평가

```bash
# 테스트셋 평가
python $NEMO_ROOT/examples/asr/speech_to_text_eval.py \
    model_path=./experiments/Timbel-Korean-FastConformer-Streaming-Pilot/*/checkpoints/best.nemo \
    dataset_manifest=./data/manifests/test_manifest_validated.json \
    output_filename=./outputs/test_predictions.json \
    batch_size=32 \
    amp=true
```

### 7.2 도메인별 평가 스크립트

**`scripts/evaluate_by_domain.py`** 생성:

```python
#!/usr/bin/env python3
"""
도메인별 (일반/전화) 성능 평가
"""

import json
import argparse
from jiwer import wer, cer
from collections import defaultdict


def load_predictions(pred_file):
    """예측 결과 로드"""
    with open(pred_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_manifest(manifest_file):
    """Manifest 로드"""
    entries = {}
    with open(manifest_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                entries[entry['audio_filepath']] = entry
    return entries


def evaluate_by_domain(predictions, manifest):
    """도메인별 평가"""
    
    domain_results = defaultdict(lambda: {'refs': [], 'hyps': []})
    
    for pred in predictions:
        audio_path = pred.get('audio_filepath', '')
        hyp = pred.get('pred_text', '')
        
        if audio_path in manifest:
            ref = manifest[audio_path].get('text', '')
            domain = manifest[audio_path].get('domain', 'unknown')
            
            domain_results[domain]['refs'].append(ref)
            domain_results[domain]['hyps'].append(hyp)
            domain_results['all']['refs'].append(ref)
            domain_results['all']['hyps'].append(hyp)
    
    # 결과 계산
    print("=" * 60)
    print(f"{'Domain':<15} {'Count':<10} {'WER (%)':<12} {'CER (%)':<12}")
    print("=" * 60)
    
    for domain, data in sorted(domain_results.items()):
        refs = data['refs']
        hyps = data['hyps']
        
        if refs:
            domain_wer = wer(refs, hyps) * 100
            domain_cer = cer(refs, hyps) * 100
            
            print(f"{domain:<15} {len(refs):<10} {domain_wer:<12.2f} {domain_cer:<12.2f}")
    
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True, help='Predictions JSON file')
    parser.add_argument('--manifest', required=True, help='Original manifest with domain info')
    
    args = parser.parse_args()
    
    predictions = load_predictions(args.predictions)
    manifest = load_manifest(args.manifest)
    
    evaluate_by_domain(predictions, manifest)
```

실행:

```bash
python scripts/evaluate_by_domain.py \
    --predictions ./outputs/test_predictions.json \
    --manifest ./data/manifests/test_manifest_validated.json
```

---

## 8. 스트리밍 추론 테스트

### 8.1 Cache-Aware 스트리밍 시뮬레이션

```bash
# NeMo 스트리밍 시뮬레이션 스크립트 사용
python $NEMO_ROOT/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
    model_path=./experiments/Timbel-Korean-FastConformer-Streaming-Pilot/*/checkpoints/best.nemo \
    dataset_manifest=./data/manifests/test_manifest_validated.json \
    output_path=./outputs/streaming_results \
    batch_size=16 \
    att_context_size="[70,13]" \
    simulate_cache_aware_streaming=true
```

### 8.2 다양한 지연 시간 테스트

```bash
# 0ms lookahead (가장 빠름, 정확도 낮음)
python $NEMO_ROOT/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
    model_path=./experiments/.../best.nemo \
    dataset_manifest=./data/manifests/test_manifest_validated.json \
    output_path=./outputs/streaming_0ms \
    att_context_size="[70,0]"

# 80ms lookahead
python ... att_context_size="[70,1]"

# 480ms lookahead
python ... att_context_size="[70,6]"

# 1040ms lookahead (기본, 가장 정확)
python ... att_context_size="[70,13]"
```

### 8.3 실시간 스트리밍 데모

**`scripts/realtime_streaming_demo.py`** 생성:

```python
#!/usr/bin/env python3
"""
실시간 스트리밍 ASR 데모
마이크 입력 → 실시간 전사
"""

import numpy as np
import pyaudio
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf, open_dict
import threading
import queue
import time


class RealtimeASR:
    def __init__(self, model_path, chunk_duration_ms=1040):
        # 모델 로드
        print(f"Loading model: {model_path}")
        self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(model_path)
        self.model.eval()
        self.model.freeze()
        
        # 스트리밍 설정
        self.sample_rate = 16000
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(self.sample_rate * chunk_duration_ms / 1000)
        
        # 오디오 버퍼
        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_queue = queue.Queue()
        
        # 상태
        self.is_running = False
        
    def start_microphone(self):
        """마이크 입력 시작"""
        self.is_running = True
        
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        stream.start_stream()
        print("Microphone started. Speak now...")
        print("-" * 50)
        
        try:
            while self.is_running:
                self._process_audio()
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """오디오 콜백"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (None, pyaudio.paContinue)
    
    def _process_audio(self):
        """오디오 처리 및 전사"""
        try:
            audio_chunk = self.audio_queue.get(timeout=0.1)
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
            
            # 충분한 오디오가 모이면 전사
            if len(self.audio_buffer) >= self.chunk_size:
                # 전사
                transcript = self.model.transcribe(
                    [self.audio_buffer[:self.chunk_size]],
                    batch_size=1
                )[0]
                
                if transcript.strip():
                    print(f"\r[{time.strftime('%H:%M:%S')}] {transcript}", end="", flush=True)
                
                # 버퍼 관리 (오버랩 유지)
                overlap = int(self.chunk_size * 0.5)
                self.audio_buffer = self.audio_buffer[self.chunk_size - overlap:]
                
        except queue.Empty:
            pass
    
    def stop(self):
        """중지"""
        self.is_running = False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to .nemo model')
    parser.add_argument('--chunk_ms', type=int, default=1040, help='Chunk duration in ms')
    args = parser.parse_args()
    
    asr = RealtimeASR(args.model, args.chunk_ms)
    asr.start_microphone()


if __name__ == '__main__':
    main()
```

실행:

```bash
pip install pyaudio

python scripts/realtime_streaming_demo.py \
    --model ./experiments/.../best.nemo \
    --chunk_ms 1040
```

---

## 9. 성능 분석

### 9.1 성능 분석 스크립트

**`scripts/analyze_results.py`** 생성:

```python
#!/usr/bin/env python3
"""
훈련 결과 분석 및 리포트 생성
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from jiwer import wer, cer


def analyze_training_logs(exp_dir):
    """훈련 로그 분석"""
    # TensorBoard 로그에서 메트릭 추출 (간소화 버전)
    print(f"Analyzing: {exp_dir}")
    
    # 체크포인트 목록
    ckpt_dir = Path(exp_dir) / 'checkpoints'
    if ckpt_dir.exists():
        checkpoints = list(ckpt_dir.glob('*.nemo'))
        print(f"Found {len(checkpoints)} checkpoints")
        for ckpt in sorted(checkpoints)[-5:]:
            print(f"  - {ckpt.name}")


def analyze_predictions(pred_file, manifest_file):
    """예측 결과 상세 분석"""
    
    # 데이터 로드
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    manifest = {}
    with open(manifest_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            manifest[entry['audio_filepath']] = entry
    
    # 분석
    results = []
    for pred in predictions:
        audio_path = pred.get('audio_filepath', '')
        hyp = pred.get('pred_text', '')
        
        if audio_path in manifest:
            ref = manifest[audio_path].get('text', '')
            domain = manifest[audio_path].get('domain', 'unknown')
            duration = manifest[audio_path].get('duration', 0)
            
            sample_wer = wer(ref, hyp) if ref else 0
            sample_cer = cer(ref, hyp) if ref else 0
            
            results.append({
                'audio_filepath': audio_path,
                'reference': ref,
                'hypothesis': hyp,
                'domain': domain,
                'duration': duration,
                'wer': sample_wer,
                'cer': sample_cer
            })
    
    df = pd.DataFrame(results)
    
    # 통계
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    print(f"Total samples: {len(df)}")
    print(f"Total duration: {df['duration'].sum() / 3600:.2f} hours")
    print(f"Average WER: {df['wer'].mean() * 100:.2f}%")
    print(f"Average CER: {df['cer'].mean() * 100:.2f}%")
    
    # 도메인별
    print("\n" + "=" * 70)
    print("BY DOMAIN")
    print("=" * 70)
    for domain in df['domain'].unique():
        domain_df = df[df['domain'] == domain]
        print(f"\n{domain}:")
        print(f"  Samples: {len(domain_df)}")
        print(f"  Duration: {domain_df['duration'].sum() / 3600:.2f}h")
        print(f"  WER: {domain_df['wer'].mean() * 100:.2f}%")
        print(f"  CER: {domain_df['cer'].mean() * 100:.2f}%")
    
    # 에러 분석
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS (Top 10 worst samples)")
    print("=" * 70)
    worst = df.nlargest(10, 'wer')
    for _, row in worst.iterrows():
        print(f"\nWER: {row['wer']*100:.1f}% | Domain: {row['domain']}")
        print(f"  REF: {row['reference'][:80]}...")
        print(f"  HYP: {row['hypothesis'][:80]}...")
    
    # Duration별 성능
    print("\n" + "=" * 70)
    print("BY DURATION")
    print("=" * 70)
    df['duration_bin'] = pd.cut(df['duration'], bins=[0, 2, 5, 10, 20, 100], 
                                 labels=['0-2s', '2-5s', '5-10s', '10-20s', '20s+'])
    duration_stats = df.groupby('duration_bin').agg({
        'wer': 'mean',
        'cer': 'mean',
        'duration': 'count'
    }).rename(columns={'duration': 'count'})
    print(duration_stats.to_string())
    
    return df


def generate_report(df, output_path):
    """리포트 생성"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. WER 분포
    axes[0, 0].hist(df['wer'] * 100, bins=50, edgecolor='black')
    axes[0, 0].set_xlabel('WER (%)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('WER Distribution')
    axes[0, 0].axvline(df['wer'].mean() * 100, color='r', linestyle='--', 
                       label=f'Mean: {df["wer"].mean()*100:.1f}%')
    axes[0, 0].legend()
    
    # 2. 도메인별 WER
    domain_wer = df.groupby('domain')['wer'].mean() * 100
    axes[0, 1].bar(domain_wer.index, domain_wer.values)
    axes[0, 1].set_xlabel('Domain')
    axes[0, 1].set_ylabel('WER (%)')
    axes[0, 1].set_title('WER by Domain')
    
    # 3. Duration vs WER
    axes[1, 0].scatter(df['duration'], df['wer'] * 100, alpha=0.5, s=10)
    axes[1, 0].set_xlabel('Duration (s)')
    axes[1, 0].set_ylabel('WER (%)')
    axes[1, 0].set_title('Duration vs WER')
    
    # 4. CER 분포
    axes[1, 1].hist(df['cer'] * 100, bins=50, edgecolor='black')
    axes[1, 1].set_xlabel('CER (%)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('CER Distribution')
    axes[1, 1].axvline(df['cer'].mean() * 100, color='r', linestyle='--',
                       label=f'Mean: {df["cer"].mean()*100:.1f}%')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nReport saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', default='./outputs/analysis_report.png')
    
    args = parser.parse_args()
    
    df = analyze_predictions(args.predictions, args.manifest)
    generate_report(df, args.output)
```

실행:

```bash
python scripts/analyze_results.py \
    --predictions ./outputs/test_predictions.json \
    --manifest ./data/manifests/test_manifest_validated.json \
    --output ./outputs/analysis_report.png
```

---

## 10. Go/No-Go 결정

### 10.1 성공 기준 체크리스트

```
┌─────────────────────────────────────────────────────────────┐
│                    GO / NO-GO 체크리스트                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ✅ 기술 검증                                                │
│     □ 훈련 수렴 확인 (loss 감소)                             │
│     □ 메모리/GPU 활용 정상                                   │
│     □ 체크포인트 저장/로드 정상                              │
│                                                             │
│  ✅ 성능 기준                                                │
│     □ 일반 음성 CER < 10%                                   │
│     □ 전화 음성 CER < 15%                                   │
│     □ 전체 WER < 15%                                        │
│                                                             │
│  ✅ 스트리밍 검증                                            │
│     □ 스트리밍 추론 정상 동작                                │
│     □ 배치 ↔ 스트리밍 결과 일치                              │
│     □ 지연 시간 목표 달성 (< 1.5초)                          │
│                                                             │
│  ✅ 스케일링 예측                                            │
│     □ 훈련 시간 선형 증가 확인                               │
│     □ 데이터 증가 시 성능 향상 예측 가능                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 결정 기준

| 결과 | 조건 | 다음 단계 |
|------|------|-----------|
| **GO** | 모든 기준 충족 | 5000시간 → 38000시간 스케일업 |
| **CONDITIONAL GO** | 대부분 충족, 일부 미달 | 하이퍼파라미터 조정 후 재시도 |
| **NO-GO** | 다수 기준 미달 | 원인 분석 및 전략 재검토 |

### 10.3 스케일업 성능 예측

```
1000시간 파일럿 결과를 바탕으로 38000시간 예측:

일반적인 스케일링 법칙:
- WER ∝ (Data)^(-0.3 ~ -0.5)
- 데이터 10배 증가 → WER 50-70% 감소

예측:
┌──────────────┬───────────┬───────────┬───────────┐
│ 데이터       │ 1000시간  │ 5000시간  │ 38000시간 │
├──────────────┼───────────┼───────────┼───────────┤
│ 일반 CER     │ ~7%       │ ~4-5%     │ ~2-3%     │
│ 전화 CER     │ ~12%      │ ~7-9%     │ ~4-6%     │
│ 훈련 시간    │ 2-4일     │ 1주       │ 3-4주     │
│ GPU 비용     │ $300      │ $1,500    │ $10,000+  │
└──────────────┴───────────┴───────────┴───────────┘
```

---

## 11. 트러블슈팅

### 11.1 일반적인 문제

#### CUDA Out of Memory

```bash
# 해결 방법
1. batch_size 감소: model.train_ds.batch_size=8
2. accumulate_grad_batches 증가: trainer.accumulate_grad_batches=8
3. precision 변경: trainer.precision=16-mixed
4. max_duration 감소: model.train_ds.max_duration=15.0
```

#### 훈련 수렴 안됨

```bash
# 해결 방법
1. learning rate 조정: model.optim.lr=0.0005
2. warmup steps 증가: model.optim.sched.warmup_steps=10000
3. gradient clipping 조정: trainer.gradient_clip_val=0.5
4. 데이터 정규화 확인
```

#### 스트리밍 성능 저하

```bash
# 해결 방법
1. FastEmit lambda 조정: model.loss.warprnnt_numba_kwargs.fastemit_lambda=0.005
2. Context size 확인: att_context_size가 훈련/추론 일치하는지
3. Multi-latency 훈련 고려
```

### 11.2 로그 확인

```bash
# 훈련 로그
tail -f ./experiments/*/nemo_log*.txt

# GPU 사용량
watch -n 1 nvidia-smi

# 디스크 사용량
df -h
```

---

## 12. 스케일업 계획

### 12.1 파일럿 성공 시 다음 단계

```
Phase 1: 파일럿 (완료)
├── 1000시간
├── 1-2 GPU
└── 1주

Phase 2: 중간 실험 (선택)
├── 5000시간
├── 2-4 GPU
└── 1-2주

Phase 3: 풀 스케일
├── 38000시간
├── 8 GPU
└── 3-4주
```

### 12.2 풀 스케일 Config 변경사항

```yaml
# 38000시간 풀 스케일 변경사항

model:
  encoder:
    n_layers: 24              # 17 → 24 (Large)
    
  tokenizer:
    vocab_size: 4096          # 1024 → 4096
    
  train_ds:
    batch_size: 32            # GPU당
    # Tarred dataset 사용 (I/O 최적화)
    is_tarred: true
    tarred_audio_filepaths: ...
    
  optim:
    sched:
      warmup_steps: 25000     # 5000 → 25000

trainer:
  devices: 8
  num_nodes: 1               # 또는 멀티 노드
  max_epochs: 500
  
  # DDP 최적화
  strategy: ddp_find_unused_parameters_true
```

---

## 📎 부록

### A. 유용한 명령어 모음

```bash
# 모델 정보 확인
python -c "
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from('model.nemo')
print(model.summarize())
"

# 체크포인트 변환
python $NEMO_ROOT/scripts/checkpoint_converters/convert_nemo_to_onnx.py \
    --nemo_file model.nemo \
    --onnx_file model.onnx

# Manifest 통계
python -c "
import json
durations = []
with open('manifest.json') as f:
    for line in f:
        durations.append(json.loads(line)['duration'])
print(f'Total: {sum(durations)/3600:.1f}h, Count: {len(durations)}')
"
```

### B. 참고 자료

- [NeMo ASR Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html)
- [Cache-Aware Streaming Paper](https://arxiv.org/abs/2312.17279)
- [FastConformer Paper](https://arxiv.org/abs/2305.05084)
- [NeMo GitHub](https://github.com/NVIDIA/NeMo)

---

**Document Version**: 1.0  
**Last Updated**: 2025-02-05  
**Author**: Timbel AI Team