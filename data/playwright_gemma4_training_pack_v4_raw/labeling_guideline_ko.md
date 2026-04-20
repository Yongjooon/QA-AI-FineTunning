# Labeling Guideline (v4 Raw)

## 목표
raw Playwright artifact bundle을 읽고, 근거 기반으로 Playwright 실행 JSON을 생성한다.

## 허용 step
- goto
- fill
- click
- expect

## 허용 assertion
- toHaveURL
- toBeVisible
- toHaveValue

## locator 우선순위
1. role
2. text
3. label
4. testId
5. locator(selectorHint fallback)

## 금지
- 입력에 없는 URL 기대값
- 입력에 없는 텍스트 기대값
- custom JS
- waitForTimeout
- 추정 기반 성공 메시지 생성

## trigger 규칙
trigger scenario는 아래 둘 중 하나가 있어야 한다.
- trigger result에서 `newTexts`
- raw bundle에 명시된 동적 노출 근거

## form 규칙
form control에 name/label이 없으면 selectorHint locator fallback 사용 가능
