# Playwright Gemma4 Training Pack v4 (Raw-based)

이 데이터셋은 `canonical page summary -> scenario JSON` 학습용이 아니라,
`raw Playwright artifact bundle -> executable scenario JSON` 학습용입니다.

## 핵심 차이
- v3: 사람이 많이 정리한 요약 입력
- v4: 원본 산출물 구조를 보존한 semi-raw bundle 입력

## 입력 특징
각 record의 input에는 다음이 들어갑니다.
- `source_zip`: 어떤 원본 압축에서 추출했는지
- `source_files_present`: 어떤 raw 파일들이 근거로 사용되었는지
- `page.url`, `page.title`, `page_type_hint`
- `visible_texts`
- `links`
- `form_controls`
- `trigger_candidates`

즉, 모델은 더 이상 단일 요약문이 아니라 원본 관찰값 묶음을 읽고 시나리오를 만들어야 합니다.

## 출력 특징
출력은 Playwright 실행기 친화 JSON입니다.
- `scenarioId`
- `title`
- `startUrl`
- `steps`
  - `goto`
  - `fill`
  - `click`
  - `expect`

## Grounding 원칙
이 데이터셋은 아래 규칙을 강제합니다.
1. 입력에 없는 URL은 `toHaveURL`로 출력하지 않음
2. 입력에 없는 텍스트는 `toBeVisible`로 출력하지 않음
3. form locator는 raw evidence에 있는 role / selectorHint / id만 사용
4. trigger 시나리오는 trigger result에 `newTexts`가 있을 때만 생성

## 데이터 크기
- train: 94
- eval: 16

## 용도
- Gemma4 QLoRA 초기 실험
- raw bundle 기반 실행 JSON 생성
- 이후 파이프라인:
  raw zip -> bundle builder -> Gemma4 -> Playwright runner

## 한계
- zip 바이너리 자체를 읽는 데이터셋은 아님
- screenshot 이미지는 포함하지 않음
- 현재는 텍스트 기반 raw bundle만 사용
- policy 페이지처럼 evidence가 적은 샘플은 open/URL 확인 위주
