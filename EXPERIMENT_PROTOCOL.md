# Recursive Conclusion Lab - 評価指標つき実験プロトコル

## 1. 目的

この実験では、会話中に数ターンごとに「この対話が最終的にどんな結論へ向かっているか」を推定する `conclusion probe` を挿入したとき、応答軌道がどう変わるかを比較する。特に次の2点を切り分けて観測する。

1. **観測効果** - probe を挿すだけで、返信内容にどれだけ影響が漏れ出るか
2. **誘導効果** - `soft_steer` で probe を次ターンへ注入したとき、収束・記憶保持・過剰ロックインがどう変化するか

加えて、`recent_window_messages` を絞った条件で `memory capsule` が文脈保持にどれだけ効くかを測る。

## 2. 主仮説

- **H1: observe でも leakage は起きる**  
  `observe` では probe をユーザーへ見せず reply system にも注入しないが、side-channel の生成が返信に影響するなら `probe_reply_overlap` は 0 より明確に大きくなる。

- **H2: soft_steer は収束を強める**  
  `soft_steer` では `probe_reply_overlap` と `conclusion_stability` が上がりやすい。

- **H3: soft_steer は late-turn redirection に弱い**  
  ユーザーが終盤で目標を変えたとき、以前の probe 仮説へ引っ張られて追従性が落ちる可能性がある。

- **H4: memory capsule は小ウィンドウ条件で有利**  
  `window=4` などの厳しい条件で、初期制約の再現率が上がるはず。

## 3. 最小実験行列

まずは以下の 4 条件を **最小コア比較** とする。

| Condition | memory_every | conclusion_every | conclusion_mode | ねらい |
|---|---:|---:|---|---|
| A. Baseline | 0 | 0 | observe | 素の対話 |
| B. Memory only | 2 or 3 | 0 | observe | memory capsule 単体の効果 |
| C. Observe | 2 or 3 | 2 or 3 | observe | probe を挿すだけの観測効果 |
| D. Soft steer | 2 or 3 | 2 or 3 | soft_steer | probe 注入の誘導効果 |

推奨パラメータ:

- **狭い文脈条件**: `window=4`
- **通常条件**: `window=8`
- **memory cadence**: `memory_every=2`
- **probe cadence**: `conclusion_every=2`
- **steer strength**: `--conclusion-steer-strength medium|strong`（`soft_steer` 時のみ。差が弱いときは `strong` で効果量を上げる）
- **steer injection**: `--conclusion-steer-injection full|conclusion_line`（`soft_steer` 時のみ。`conclusion_line` は注入ノイズを減らして比較を安定させやすい）
- **reply temperature**: 0.0 または 0.2
- **probe temperature**: 0.0 固定

## 4. ベンチマーク会話ファミリー

### Family 1: Convergent planning

目的: `soft_steer` が計画系タスクの収束を強めるかを見る。

期待:
- `soft_steer` は final answer をより早く構造化する
- ただし premature narrowing が起きると代替案が減る

推奨スクリプト: `protocol_scripts/convergent_protocol.json`

### Family 2: Context retention under compression

目的: 小ウィンドウ条件で early-turn anchor を保持できるかを見る。

期待:
- memory なしでは後半で exact anchor を落としやすい
- memory ありでは final answer の anchor 再現率が上がる

推奨スクリプト: `protocol_scripts/context_retention.json`

### Family 3: Late redirection / anti-lock-in

目的: 終盤でユーザーが方針転換したとき、以前の結論仮説に固着しないかを見る。

期待:
- `soft_steer` は収束が強い一方、late redirection 追従性で不利になる可能性がある
- `observe` は追従しやすいが収束の速さでは負けるかもしれない

推奨スクリプト: `protocol_scripts/late_redirection.json`

## 5. 指標

### 5.1 既存ログから直接取れる指標

1. **probe_reply_overlap**  
   既に実装済み。reply が直前 probe 仮説をどれだけ取り込んだかの近似。

2. **avg_probe_confidence**  
   `CONCLUSION / CONFIDENCE / EVIDENCE` 形式から confidence を抽出し平均する。

3. **conclusion_stability**  
   連続する probe 同士の lexical overlap。高すぎると硬直、低すぎると不安定の可能性。

4. **last_turn_alignment**  
   最後の user turn と最後の assistant reply の lexical overlap。現在の要求への追従性の粗い近似。

5. **final_probe_reply_overlap**  
   最後の probe 仮説と最後の reply の overlap。最終ターンで仮説がどれだけ reply を支配したかを見る。

6. **usage-based efficiency**  
   `usage` が取れる provider では、turn あたり input/output token を比較する。

7. **conclusion_line_mention_rate / avg_conclusion_line_mention_delay_turns**  
   conclusion probe が出した `CONCLUSION:` 行が、どれくらいの遅延で assistant reply に「言及」されるか（観測ベース）。

8. **conclusion_any_mention_rate**  
   `conclusion_line` の lexical overlap だけでなく、probe が出した `keywords` による言及検出も含めた mention rate。

9. **conclusion_plan_within_window_rate**  
   probe が出す observe-only の言及プラン（`mention_delay_min_turns` / `mention_delay_max_turns`）に対して、実際の言及遅延がウィンドウ内に収まった割合。

### 5.2 スクリプト評価 spec から計算する指標

`script.json` に `evaluation` セクションを足し、以下を計算する。

1. **final_required_keyword_coverage**  
   final reply に必須キーワードがどれだけ入ったか。

2. **conversation_required_keyword_coverage**  
   会話全体の assistant reply 群に必須キーワードがどれだけ現れたか。

3. **final_forbidden_keyword_hits**  
   final reply に不要キーワードが何個入ったか。late redirection で古い方針を引きずっていないかの近似。

### 5.3 推奨の補助評価

lexical 指標だけだと意味的な良し悪しを取りこぼす。最低でも以下のどちらかを追加するとよい。

- **人手評価 3 項目**  
  1. 現在のユーザー目標に答えているか  
  2. 初期制約を保持しているか  
  3. 以前の probe 仮説に引っ張られすぎていないか

- **LLM-as-judge**  
  ただし judge model を使う場合は、被評価モデルと別系列にする。例: 被評価が OpenAI 系なら judge は Anthropic 系や Gemini 系も併用する。

## 6. まず回すべき順序

### Phase 0: ローカル smoke test

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/convergent_protocol.json \
  --providers dummy=dummy-v1 \
  --window 8 \
  --memory-every 2 \
  --conclusion-every 2 \
  --conclusion-mode observe \
  --out-dir runs/smoke_observe
```

### Phase 1: 単一 provider で条件差を確認

同一 provider / 同一 model で 4 条件 A-D を回し、指標がちゃんと分離するかを見る。

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/context_retention.json \
  --providers openai=<model_id> \
  --window 4 \
  --memory-every 0 \
  --conclusion-every 0 \
  --conclusion-mode observe \
  --out-dir runs/openai_A_baseline
```

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/context_retention.json \
  --providers openai=<model_id> \
  --window 4 \
  --memory-every 2 \
  --conclusion-every 0 \
  --conclusion-mode observe \
  --out-dir runs/openai_B_memory_only
```

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/context_retention.json \
  --providers openai=<model_id> \
  --window 4 \
  --memory-every 2 \
  --conclusion-every 2 \
  --conclusion-mode observe \
  --out-dir runs/openai_C_observe
```

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/context_retention.json \
  --providers openai=<model_id> \
  --window 4 \
  --memory-every 2 \
  --conclusion-every 2 \
  --conclusion-mode soft_steer \
  --out-dir runs/openai_D_soft_steer
```

### Phase 2: プロバイダ横断比較

Phase 1 で差が出た設定だけ、OpenAI / Anthropic / Mistral / Gemini / HF へ拡張する。

### Phase 3: 反事実検証

最も差が大きかった script family に対し、

- `window=4` vs `window=8`
- `temperature=0.0` vs `temperature=0.2`
- `observe` vs `soft_steer`
- `--conclusion-steer-strength medium` vs `--conclusion-steer-strength strong`
- `--conclusion-steer-injection full` vs `--conclusion-steer-injection conclusion_line`

を切り替えて再確認する。

## 7. 解釈ガイド

- **probe_reply_overlap 高 / final_goal_coverage 高**  
  probe 仮説がうまく reply の構造化を助けている可能性

- **probe_reply_overlap 高 / late_redirection 失敗**  
  仮説が強すぎて lock-in している可能性

- **conclusion_stability 極端に高い / final quality 低い**  
  早すぎる premature convergence の疑い

- **memory only で recall 改善 / observe でさらに悪化**  
  probe が意図せず reply を汚染している可能性

## 8. 最初の推奨セット

いきなり全部回さず、次の 6 実行から始めるとよい。

1. `convergent_protocol.json` x `observe` x `window=8`
2. `convergent_protocol.json` x `soft_steer` x `window=8`
3. `context_retention.json` x `baseline` x `window=4`
4. `context_retention.json` x `memory_only` x `window=4`
5. `late_redirection.json` x `observe` x `window=8`
6. `late_redirection.json` x `soft_steer` x `window=8`

これで、

- 収束が強まるか
- 記憶保持が改善するか
- 過剰ロックインが出るか

の 3 軸が一通り見える。

## 9. 成功パターンと失敗パターン

### 成功パターン
- `memory_only` が context retention を改善
- `observe` が軽い leakage に留まる
- `soft_steer` が convergent planning では有利
- ただし late redirection では `observe` が勝つ

### 失敗パターン
- `observe` の時点で overlap が高すぎる
- `soft_steer` で final answer が毎回似すぎる
- redirection 後も obsolete goal を繰り返す
- memory capsule が stylistic summary になり facts を落とす

## 10. 実用上の結論

最終的に知りたいのは「probe を観測装置として使うべきか、制御装置として使うべきか」である。

- **観測器として十分** なら `observe` を採用
- **収束促進に価値** があるなら `soft_steer` を planning タスクに限定採用
- **goal shift が多い運用** なら `soft_steer` は危険なので抑制する

---

補助スクリプト `analyze_runs.py` は、このプロトコルに沿って JSONL ログから指標を集計するために用意してある。
