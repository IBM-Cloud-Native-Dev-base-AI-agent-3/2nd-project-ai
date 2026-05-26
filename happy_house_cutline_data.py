import pandas as pd
import re

def normalize_elig(x):
    x = str(x).strip()
    if '대학' in x:                  return '대학생'
    if '주거급여' in x or '수급' in x: return '주거급여'
    if '고령' in x:                  return '고령자'
    if '신혼' in x:                  return '신혼부부'
    if '청년' in x:                  return '청년'
    if '일반' in x:                  return '일반가구'
    return x

def parse_happy_md(md_path, stage):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    rows = []
    current_year = None
    current_phase = '1차'

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # ★ | 체크 전에 연도/차수 먼저 추출
        year_match = re.search(r'(20\d{2})\s*년?\s*[-–]?\s*(\d)\s*차', line)
        if year_match:
            current_year = float(year_match.group(1))
            current_phase = f'{year_match.group(2)}차'

        # 테이블 행만 처리
        if '|' not in line:
            continue

        cells = [c.strip() for c in line.split('|')]
        cells = [c for c in cells if c]

        if len(cells) < 3:
            continue

        if '단지명' in cells[0] or '---' in cells[0]:
            continue

        complex_name = cells[0] if len(cells) > 0 else ''
        supply_cell  = cells[1] if len(cells) > 1 else ''
        rank_cell    = cells[2] if len(cells) > 2 else ''
        score_cell   = cells[3] if len(cells) > 3 else ''
        size_cell    = cells[4] if len(cells) > 4 else ''

        if not re.search(r'[가-힣]{2,}', complex_name):
            continue

        elig_match = re.search(r'행복주택\((.+?)\)', supply_cell)
        if not elig_match:
            continue

        if any(w in score_cell for w in ['추첨', '전원', '-', '미달']):
            continue

        score_match = re.search(r'(\d+(?:\.\d+)?)\s*점', score_cell)
        if not score_match:
            continue
        score = float(score_match.group(1))
        if score < 1 or score > 9:
            continue

        rank_match = re.search(r'(\d+)\s*순위', rank_cell)
        rank = int(rank_match.group(1)) if rank_match else 1

        size = size_cell if re.match(r'^\d+S?$', size_cell) else '알수없음'

        rows.append({
            '공고연도':     current_year,   # ← None 아닌 실제 연도
            '공고차수':     current_phase,
            '단지명':       complex_name,
            '공급유형':     size,
            '신청자격':     normalize_elig(elig_match.group(1)),
            '당첨순위':     rank,
            '심사단계':     stage,
            '커트라인점수':  score,
        })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    df_doc   = parse_happy_md('parsed_document_cutline.md', stage='서류')
    df_final = parse_happy_md('parsed_final_cutline.md',    stage='최종')

    print(f'서류심사 파싱: {len(df_doc)}행')
    print(f'최종당첨 파싱: {len(df_final)}행')

    df_new = pd.concat([df_doc, df_final], ignore_index=True)
    df_new = df_new.drop_duplicates().reset_index(drop=True)

    print(f'\n신규 합계: {len(df_new)}행')
    print('\n연도별 분포:')
    print(df_new['공고연도'].value_counts().sort_index())
    print('\n신청자격 분포:')
    print(df_new['신청자격'].value_counts())

    df_old = pd.read_csv('cutline_data.csv', encoding='utf-8-sig')
    print(f'\n기존 데이터: {len(df_old)}행')

    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all = df_all.drop_duplicates().reset_index(drop=True)
    df_all.to_csv('cutline_data_v2.csv', index=False, encoding='utf-8-sig')
    print(f'저장 완료: cutline_data_v2.csv ({len(df_all)}행)')