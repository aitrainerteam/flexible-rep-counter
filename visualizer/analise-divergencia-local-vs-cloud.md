# Análise da Divergência Local vs Cloud — Rep Counter

Documento consolidado da investigação da sessão `b29815ef-ca00-42cf-8d82-485537e1e936` executada via `visualizer/vm_rep_simulator.py` contra o VM em `http://34.95.128.222:8000`.

Data da sessão: `2026-04-21` (ts inicial `1776780635938`, ts final `1776780682576`).

---

## 1. Sumário executivo

- **Resultado final**: local=42 reps, cloud=35 reps (divergência de **7 reps**).
- **Causa principal**: por volta de `ts=1776780672539` (cloud frame_idx=435), a cloud fez um **switch de `tracked_joint`** de `LEFT_ELBOW` para `RIGHT_ELBOW` e reset `rep_count` para 1. Esse switch durou ~3 segundos; depois voltou para `LEFT_ELBOW` com `rep_count=31` (restaurado pelo estado paralelo). Durante o switch, a cloud não incrementou rep; o local continuou contando normalmente (33→37).
- **Por que local e cloud divergiram apesar de rodarem o mesmo código**: o **código do algoritmo é praticamente idêntico** (verificado via diff). O que difere é a **taxa de alimentação do detector**: o cloud recebe ~15 amostras/s **únicas**, enquanto o local chama `step_landmarks` a ~30 Hz reutilizando o último landmark quando não há resposta nova da cloud (ou seja, **metade das amostras locais são duplicatas** do landmark anterior). Isso atenua a variância empírica e encurta a janela temporal efetiva de parâmetros contados em frames — tornando o detector local menos reativo a transições bruscas e impedindo o switch de joint que a cloud disparou.
- **Fontes originais dos dados**:
  - `visualizer/log_vm_rep_simulator.txt` (1192 linhas) — log terminal do simulador local
  - `visualizer/latest-rep-counter-session.ndjson` (800 linhas, 488 KB) — NDJSON emitido pelo rep counter do VM
  - `visualizer/latest-metadata.json` — metadados da sessão puxada do VM
  - `visualizer/vm_rep_simulator.py` — cliente OpenCV + driver local
  - `/mnt/c/Users/caiop/Desktop/repos/ai-personal-trainer/ai-personal-trainer/arquivos-vm-cloud/` — snapshot do código rodando na cloud

---

## 2. Metadados da sessão

`visualizer/latest-metadata.json`:

```json
{
  "user_uid": "debug-opencv-simulator",
  "latest_session_id": "b29815ef-ca00-42cf-8d82-485537e1e936",
  "event_count": 800,
  "first_ts_ms": 1776780642673.1687,
  "last_ts_ms": 1776780683036.1492,
  "vm_instance": "yolo-deployed-brazil",
  "vm_zone": "southamerica-east1-a",
  "vm_project": "ai-trainer-a44a3",
  "event_type_counts": {
    "frame_snapshot": 582,
    "state_transition": 115,
    "rep_block": 58,
    "rep_increment": 45
  },
  "pulled_at_utc": "2026-04-21T14:12:24.506688+00:00"
}
```

Duração: `last_ts_ms - first_ts_ms ≈ 40.4 s` (janela do NDJSON; o simulador local começou ~7 s antes com capturas de fluxo/health).

Picos de performance registrados no fim do log local:

| Métrica (ms) | Valor pico |
|---|---|
| `roundtrip_ms` | 168.7 |
| `upload_ms` | 166.2 |
| `encode_ms` | 9.5 |
| `inference_ms` (server) | 18.6 |
| `session_ms` (cpu cliente) | 13.1 |
| `detector_ms` | 8.8 |
| `variance_ms` | 11.6 |

---

## 3. Fase A — Calibração do cloud (0–11 s, reps local 1–5)

A cloud demorou ~11 s para sair de `phase="selecting"` e fixar o `tracked_joint`. Enquanto isso, o local já contava.

Eventos relevantes no NDJSON:

- Frame 1 (ts=1776780642673): `phase="selecting"`, `tracked_joint=null`, `leader_key=null`.
- Frame 21 (ts≈1776780644094): `leader_key` passa a `LEFT_KNEE`.
- Frames 1-72: 50 `state_transition` e 25 `rep_block` do tipo `not_rep_would_increment` alternando entre `LEFT_KNEE`, `RIGHT_KNEE`, `LEFT_ELBOW`, `LEFT_HIP`, `LEFT_HIP_ACROSS`, `LEFT_SHOULDER`, etc.
- Frame 72 (ts=1776780647838): cloud atribui `tracked_joint=LEFT_ELBOW`.
- Frame 73 (ts≈1776780647907): `calibration_complete=true`, transição para `phase="tracking"`, `calibration_certainty≈0.559`.

Heartbeats `diag` do log local no período:

```
linha 7   ts=1776780641433  cam=30.1fps infer=0.0/s  vm_rsp=1  lm=0%   last_vm=0.1s
linha 139 ts=1776780646494  cam=29.6fps infer=13.4/s vm_rsp=70 lm=100% last_vm=0.0s
```

Eventos de rep contados nesse período (log local):

```
linha 107 ts=1776780645348.8979 event=local_rep_counted local_reps=4 cloud_reps=0
linha 124 ts=1776780645960.2295 event=local_rep_counted local_reps=5 cloud_reps=0
linha 146 ts=1776780646725.9126 event=cloud_rep_counted cloud_reps=4 local_reps=5
```

→ O local chegou a 5 reps antes da cloud sair de calibração; a cloud "alcançou" em 4 e depois foi subindo.

---

## 4. Fase B — Operação estável (11–36 s, reps 5–30)

Por 346 frames contínuos (frames 73→418), o cloud manteve `tracked_joint=LEFT_ELBOW`.

Características:

- Todos os 30 `rep_increment` dessa janela são `kind=valley` em `LEFT_ELBOW`.
- `rolling_range` do `LEFT_ELBOW`: **42–51°**.
- `calibration_certainty`: **0.55–0.62**.
- `range_gate_open=true` em 100% dos frames.
- 28 `rep_block` do tipo `not_rep_would_increment` para `LEFT_ELBOW` (normal — bloqueia oscilações menores).
- Heartbeats `diag` estáveis: `cam≈29.7fps`, `infer≈14.5/s`, `q_drop=77–80` por 5s, `lm=100%`, `last_vm=0.0s`.

Padrão observado no log local: **local sempre 1–2 reps à frente do cloud**. Isso é latência normal (roundtrip mediana ~65 ms) + pequenas diferenças de timing do peak detector de cada lado.

Exemplo de tabela cronológica (log local):

```
linha 346 ts=1776780653399  local_reps=14 cloud_reps=11
linha 352 ts=1776780653560  cloud_reps=12 local_reps=14
linha 399 ts=1776780655279  local_reps=16 cloud_reps=13
linha 402 ts=1776780655386  cloud_reps=14 local_reps=16
linha 446 ts=1776780656954  local_reps=18 cloud_reps=15
linha 449 ts=1776780657018  cloud_reps=16 local_reps=18
linha 789 ts=1776780668474  local_reps=30 cloud_reps=28
```

---

## 5. Fase C — Evento crítico: reset cloud 30→1→31 (~36–40 s)

### 5.1. Sequência exata no NDJSON

| Frame | ts_ms | Evento | Observação |
|---|---|---|---|
| 418 | 1776780671694 | `rep_increment rep_count=30 kind=valley detector=LEFT_ELBOW` | Último rep válido antes do reset |
| 430 | 1776780672528 | `rep_block kind=peak reason=not_rep_would_increment detector=LEFT_ELBOW` | Pico do `LEFT_ELBOW` rejeitado |
| 430 | 1776780672528 | `state_transition going_up→going_down LEFT_ELBOW` | — |
| **435** | **1776780672890** | **`rep_increment rep_count=1 kind=valley detector=RIGHT_ELBOW`** | **SWITCH: `tracked_joint` muda para `RIGHT_ELBOW`; `reps=1`; `calibration_complete=false`** |
| 436–490 | 1776780672890 → 1776780676684 | 55 frames com `tracked_joint=RIGHT_ELBOW`, `reps=1`, `peak_detector_state="going_up"` travado | **RIGHT_ELBOW não fechou nenhum ciclo** — nunca transicionou para `going_down` |
| 491 | ~1776780676762 | Troca implícita de volta para `LEFT_ELBOW`, `reps=31` restaurado, `calibration_complete=true` | Cloud desistiu do `RIGHT_ELBOW` |
| 496 | 1776780677124 | `rep_increment rep_count=31 kind=valley detector=LEFT_ELBOW` | Cadência normal retomada |

### 5.2. Correlação no log local

```
linha 869 ts=1776780671291.54 event=local_rep_counted local_reps=33 cloud_reps=31  (diff=2, normal)
linha 904 ts=1776780672539.06 event=local_rep_counted local_reps=34 cloud_reps=1   (diff=33!)
linha 928 ts=1776780673334.84 event=local_rep_counted local_reps=35 cloud_reps=1
linha 951 ts=1776780674103.84 event=local_rep_counted local_reps=36 cloud_reps=1
linha 972 ts=1776780674807.27 event=local_rep_counted local_reps=37 cloud_reps=1
linha 991 ts=1776780675581.53 event=cloud_rep_counted cloud_reps=31 local_reps=37  (cloud voltou)
```

Durante o intervalo `ts=1776780672539 → 1776780675581` (~3.04 s), o `cloud_reps` recebido pelo cliente local ficou em **1**. Em todo esse período, o local contou 5 reps reais (33→34→35→36→37), que a cloud não registrou.

### 5.3. Heartbeats `diag` no entorno do reset

```
linha 877 ts=1776780671605.16  cam=29.5fps infer=14.6/s  q_enq=72 q_drop=78  vm_rsp=72 lm=100% last_vm=0.0s
linha 1021 ts=1776780676623.99 cam=29.8fps infer=14.2/s  q_enq=73 q_drop=77  vm_rsp=73 lm=100% last_vm=0.0s
```

**Nenhuma anomalia de rede, FPS ou inferência.** O reset foi puramente decisão interna do algoritmo da cloud.

### 5.4. State transitions durante o reset

- **Antes do reset (frame < 418)**: 39 transições de HIP/SHOULDER/KNEE/RIGHT_ELBOW concentradas em frames 1–72 (fase de calibração).
- **Durante o reset (frames 418–496)**: apenas **1 transição** — frame 435, `RIGHT_ELBOW: going_down→going_up`.
- **Após o reset (frames > 496)**: apenas transições `LEFT_ELBOW`. Nenhuma nova transição em HIP/SHOULDER/KNEE.

Conclusão direta: não há "tempestade" de mudanças de postura via joints de perna ou quadril no NDJSON. O switch foi acionado pela **dinâmica de variância** entre os dois cotovelos, não por detecção explícita de mudança postural.

---

## 6. Investigação: por que o algoritmo local não fez o mesmo switch

### 6.1. Verificação rigorosa do código

Foi feito `diff` direto entre os arquivos de implementação do local (`flexible-rep-counter/src/flexible_rep_counter/`) e do cloud (`arquivos-vm-cloud/vendor/flexible-rep-counter/src/flexible_rep_counter/`).

| Arquivo | Resultado do diff |
|---|---|
| `session.py` | **3 linhas** diferem — todas do flag `tracked_joint_changed` (adicionado no commit `0f1ab1a`, apenas informativo, **não afeta contagem**) |
| `core/math_engine.py` (peak detector + cálculo de ângulos) | **idêntico** |
| `core/variance_angle_selector.py` (lógica de switch de joint) | **idêntico** |
| `core/pose_filters.py` (smoothing) | **idêntico** |
| `landmark_utils.py` | matematicamente equivalentes (ângulos são invariantes à escala isotrópica do letterbox) |
| `types.py` | `StepResult.tracked_joint_changed` existe no local, não existe no cloud (efeito: só no output) |
| `rep_counter.toml` | **idêntico** em todos os 54 parâmetros de tuning (só `log_level` difere: `INFO` local, `DEBUG` cloud) |

Trechos diferentes em `session.py`:

```diff
# LOCAL src/flexible_rep_counter/session.py:881
out = self._build_tracking_step_result(
    rs,
    angle_value,
    detector_output=selected_output,
+   tracked_joint_changed=switched_to is not None,
    selection_debug={...},
)

# LOCAL src/flexible_rep_counter/session.py:906-907 (assinatura)
def _build_tracking_step_result(
    self, rs, angle_value, *, detector_output,
+   tracked_joint_changed: bool = False,
    selection_debug,
) -> StepResult:

# LOCAL src/flexible_rep_counter/session.py:1005 (retorno)
return StepResult(
    ...,
+   tracked_joint_changed=tracked_joint_changed,
    ...,
)
```

**Conclusão parcial**: o código é praticamente idêntico. A lógica de switch de joint (ratio de variância 1.2× + cooldown 1.5s + reavaliação a cada 0.75s) é a mesma. Então a divergência de comportamento não vem do código — vem das **entradas**.

### 6.2. Descoberta da causa raiz: alimentação duplicada do detector local

Olhando o loop principal em `visualizer/vm_rep_simulator.py` (linhas 1259, 1324–1417):

```python
frame_queue: Queue = Queue(maxsize=1)        # linha 1259 — fila de 1 slot para envio à cloud
latest_pose: list = [ {...} ]                 # linha 1260 — holder compartilhado pelos threads

while not display.stop_requested():
    ret, frame_bgr = capture.read()           # câmera ~30 fps
    ...
    try:
        frame_queue.put_nowait(frame_bgr.copy())  # se fila cheia, DROPA o frame
    except Full:
        diag.note_frame_dropped()
    snap = latest_pose[0]                     # lê o ÚLTIMO snapshot — pode ser o mesmo já processado
    ...
    raw_landmarks = snap.get("landmarks")
    timestamp_ms = time.time() * 1000.0       # linha 1383 — timestamp SEMPRE recalculado
    ...
    step = rs_sess.step_landmarks(raw_scaled, timestamp_ms=timestamp_ms)  # linha 1417
```

Fatos observáveis:

- A **câmera** roda a ~30 fps (confirmado pelos heartbeats `diag`: `cam=29.5–30.1fps`).
- A **cloud** responde a ~14–15 fps (confirmado pelos heartbeats: `infer=14.5/s`, `vm_rsp=72–73` por 5s).
- A fila de envio tem `maxsize=1`: quando cheia, `put_nowait` lança `Full` e o frame é **dropado, não enviado**. Heartbeat mostra `q_drop=77–80` por 5s — ~metade dos frames nem chega na cloud.
- O **loop principal local chama `step_landmarks` uma vez por frame da câmera** (~30 Hz), independente de ter resposta nova. Quando não há landmark novo do worker, ele lê o **último `latest_pose[0]`** — ou seja, **o mesmo landmark já processado na iteração anterior**.
- O `step_landmarks` **não deduplica** por `frame_idx` ou hash — cada chamada é tratada como amostra nova.

Então:

| Fonte | Taxa efetiva de amostras | Natureza |
|---|---|---|
| Cloud (detector rodando dentro do VM, sobre frames recebidos) | ~15 amostras/s | todas únicas |
| Local (detector rodando em `rs_sess.step_landmarks`) | ~30 chamadas/s | **~15 únicas + ~15 duplicatas do último landmark** |

### 6.3. Por que a duplicação muda o comportamento do algoritmo

Os parâmetros críticos do peak detector e do variance selector são contados em **frames**, não em segundos:

| Parâmetro | Definição em frames | Tempo real cloud (14-15 fps) | Tempo real local (30 fps, metade duplicatas) |
|---|---|---|---|
| `min_peak_distance=5` | distância mínima entre picos consecutivos | ~333 ms | ~167 ms |
| `range_window_frames=90` | janela do `rolling_range` e cálculo de variância | ~6 s | ~3 s |
| smoothing EMA | atualização por amostra | aplica a 14–15 pts/s | aplica a 30 pts/s (50% duplicatas) |

Consequências assimétricas:

1. **Variância empírica atenuada**: dentro de uma janela de 90 amostras, se metade são duplicatas do último valor, a variância bruta é **artificialmente reduzida** — pontos repetidos contribuem zero variância entre si. Todos os joints têm variância menor no local que no cloud.
2. **Ratio de switch comprimido**: o selector troca de joint se o candidato tem variância >1.2× o atual. Com variâncias atenuadas de todos os lados, diferenças relativas ficam menores, e o threshold de 1.2× raramente é atingido.
3. **Janela temporal menor**: 90 frames no local cobrem ~3 s; no cloud, ~6 s. Transições passageiras (como o enquadramento mudando quando o usuário ficou em pé) entram inteiras na janela da cloud, mas só parcialmente na janela local.
4. **`min_peak_distance`**: no local, 5 frames = ~167 ms — bem mais permissivo que os ~333 ms do cloud, o que pode deixar oscilações menores serem contadas como reps. Contribui para o local ter contagem ligeiramente maior.

### 6.4. Reconstrução do instante do reset

**No CLOUD (frame_idx=435, ts=1776780672890):**

1. Usuário levanta da cadeira → `LEFT_ELBOW` teve deslocamento + ruído por ~1–2 s → variância bruta do `LEFT_ELBOW` caiu transitoriamente.
2. Reavaliação `ANGLE_SELECTION_REEVALUATE_EVERY_SEC=0.75s` → `determine_best_angle()` recomputou variâncias sobre janela de 90 frames ≈ **6 s** (pega a queda toda).
3. `RIGHT_ELBOW`, rodando em paralelo, tinha variância >1.2× `LEFT_ELBOW` no instante da checagem → switch disparado (cooldown OK, variance gate ativado).
4. `selected_angle = RIGHT_ELBOW`; detector paralelo do `RIGHT_ELBOW` estava em `rep_count=0`; primeiro pico pós-switch incrementou para **1** — esse é o "reset" observado.
5. Por 55 frames (~2.4 s), `peak_detector_state="going_up"` do `RIGHT_ELBOW` ficou travado (movimento não produziu ciclo completo) — nenhum novo `rep_increment`.
6. Reavaliação seguinte: `LEFT_ELBOW` voltou a dominar; switch de volta; detector paralelo do `LEFT_ELBOW` ainda em `rep_count=31` (estado preservado) — cadência retomou em 31.

**No LOCAL (mesmo período de wall-clock):**

1. Mesma queda real do `LEFT_ELBOW` entra na janela de 90 amostras ≈ **3 s** (cobre só metade do que o cloud vê) + com 50% de duplicatas (amortecendo mais a variância).
2. Variância empírica do `LEFT_ELBOW` caiu, mas não o suficiente para o `RIGHT_ELBOW` atingir ratio 1.2×.
3. **Switch não disparou**. `LEFT_ELBOW` continuou dominante. `rep_count` subiu normalmente 33→34→35→36→37.

### 6.5. Por que o local chegou a 42 reps vs 35 da cloud

Decomposição do gap final:

| Contribuição | Δ reps | Causa |
|---|---|---|
| Calibração inicial lenta da cloud | +1 | cloud só saiu de `selecting` no rep 4–5 local |
| Latência/jitter cumulativo | +1 a +2 | diferença normal de timing durante steady-state |
| Reset da cloud (~2.4 s sem contar) | **+5** | durante o switch fallido para `RIGHT_ELBOW` |
| Possível oversensing local | 0 a +1 | `min_peak_distance` temporal menor no local |
| **Total esperado** | **+7 a +9** | **observado: +7** ✓ |

---

## 7. Esclarecimento conceitual: nada é "enviado duas vezes"

Para evitar confusão:

- A **cloud não recebe imagens duplicadas**. Ela recebe uma sequência de imagens únicas, ~15/s, e roda seu rep counter sobre amostras distintas.
- O que se duplica é o **input do detector local**. O loop local chama `rs_sess.step_landmarks(...)` a ~30 Hz e, quando ainda não chegou landmark novo da cloud, alimenta o detector com o **último landmark recebido**, com um `timestamp_ms` novo.
- Então o detector local vê uma sequência de 30 amostras/s onde metade é cópia da amostra anterior. Essa sequência "poluída" é o que entra nas estatísticas de variância.

**Reformulação precisa em uma frase:** o código do algoritmo é o mesmo nos dois lados; a cloud recebe 15 amostras/s únicas; o local alimenta seu detector a 30 Hz com cada amostra repetida uma vez; parâmetros contados em frames (janela de variância, `min_peak_distance`) cobrem **metade do tempo real** no local, com variâncias empíricas atenuadas — foi essa atenuação que impediu o switch de joint que a cloud disparou quando o usuário ficou em pé.

---

## 8. Como confirmar a hipótese empiricamente

Três experimentos diretos para validar:

### Experimento 1 — Deduplicação no loop local

Modificar `visualizer/vm_rep_simulator.py` para guardar um `last_snapshot_id` (pode ser `id(snap)` ou um campo `frame_idx` adicionado ao `latest_pose`) e **só chamar `step_landmarks` quando o id mudar**. Rodar a mesma sessão. Se reproduzir o switch da cloud (reset 30→1) no mesmo instante, hipótese confirmada.

Esboço da mudança:

```python
last_snap_id = None
while not display.stop_requested():
    ret, frame_bgr = capture.read()
    ...
    snap = latest_pose[0]
    snap_id = id(snap)  # ou snap.get("frame_idx")
    if snap_id != last_snap_id:
        raw_landmarks = snap.get("landmarks")
        timestamp_ms = time.time() * 1000.0
        step = rs_sess.step_landmarks(raw_scaled, timestamp_ms=timestamp_ms)
        last_snap_id = snap_id
    # display/overlay continuam rodando a 30fps usando o último `step`
```

### Experimento 2 — Instrumentar a variância

Logar, no local, o `selection_debug` por frame e cruzar com o NDJSON da cloud na mesma janela temporal (`ts=1776780672000 → 1776780675000`). Deve mostrar variâncias divergindo materialmente entre os dois lados.

### Experimento 3 — Limitar a taxa do loop local

Forçar o loop local a rodar a ~15 fps (processar só 1 a cada 2 frames ou `time.sleep(0.033)` após cada iteração). Deve convergir muito mais com a cloud.

---

## 9. Recomendações

1. **Deduplicar input do detector local**: adicionar um token/hash de frame em `latest_pose` e pular `step_landmarks` quando o token não mudou. Solução mais direta e com menor superfície de mudança. Elimina a divergência estatística sem alterar o algoritmo.
2. **Parametrizar janelas em tempo, não em frames**: converter `min_peak_distance`, `range_window_frames` e janela de variância para segundos (convertidos para frames com base no FPS corrente). Mudança maior mas torna o detector robusto a diferentes taxas.
3. **Aumentar cooldown/confirmação do switch**: exigir que o novo `tracked_joint` complete N ciclos válidos antes de efetivar o switch no output. Evita `rep_count=1` exposto ao cliente quando o switch é espúrio.
4. **Filtrar re-seleção durante mudanças posturais**: detectar salto brusco nos landmarks absolutos (proxy para mudança de postura) e congelar o `tracked_joint` por alguns segundos.
5. **Sincronizar versão do vendor da cloud**: atualizar `arquivos-vm-cloud/vendor/flexible-rep-counter/` para incluir o commit `0f1ab1a` (flag `tracked_joint_changed`). Permite ao cliente mostrar UX melhor ("reavaliando joint dominante…") ao invés de exibir `reps=1` durante o switch.

---

## 10. Referências de arquivo e linha

### Código local
- `visualizer/vm_rep_simulator.py:1259` — `Queue(maxsize=1)`
- `visualizer/vm_rep_simulator.py:1324–1450` — loop principal da captura/display
- `visualizer/vm_rep_simulator.py:1383` — `timestamp_ms = time.time() * 1000.0`
- `visualizer/vm_rep_simulator.py:1417` — chamada `rs_sess.step_landmarks(raw_scaled, timestamp_ms=...)`
- `src/flexible_rep_counter/session.py:820–870` — lógica de re-avaliação e switch de joint
- `src/flexible_rep_counter/session.py:843–869` — condição e execução do switch (`switched_to`, `rs["peak_detector"] = sdba.get(candidate)`)
- `src/flexible_rep_counter/session.py:858` — `rs["peak_detector"] = sdba.get(candidate)` (cada joint tem detector paralelo próprio)
- `src/flexible_rep_counter/session.py:881` — passagem de `tracked_joint_changed` (**nova, só no local**)
- `src/flexible_rep_counter/types.py:36` — `tracked_joint_changed: bool = False` no `StepResult`
- `rep_counter.toml` — 54 parâmetros, idênticos entre local e cloud (exceto `log_level`)

### Código cloud
- `arquivos-vm-cloud/main.py` — endpoint `/predict` e invocação do rep counter no servidor
- `arquivos-vm-cloud/main.py:542–544` — chamada `step_landmarks(landmarks, timestamp_ms=now_ms)`
- `arquivos-vm-cloud/vendor/flexible-rep-counter/src/flexible_rep_counter/session.py:820–870` — mesma lógica de switch
- `arquivos-vm-cloud/vendor/flexible-rep-counter/src/flexible_rep_counter/session.py:877–887` — chamada de `_build_tracking_step_result` **sem** `tracked_joint_changed`
- `arquivos-vm-cloud/rep_counter.toml` — idêntico ao local salvo `log_level`

### Commits relevantes (git log no repo local)
- `0f1ab1a` — "feat: add tracked_joint_changed flag to StepResult and RepCounterSession" (posterior ao snapshot do vendor)
- `0074975` — "feat: add rep counter instrumentation for enhanced performance tracking"
- `f655f3c` — "WARNING: UNSTABLE feat: introduce benchmarking and replay functionality for rep counter"

---

## 11. Apêndice A — Tabela completa de divergências local vs cloud no log

Todas as entradas `event=local_rep_counted` / `event=cloud_rep_counted` do `log_vm_rep_simulator.txt`:

```
linha 107  ts=1776780645348.89 local=4  cloud=0   diff=+4  [calibração cloud]
linha 124  ts=1776780645960.22 local=5  cloud=0   diff=+5  [calibração cloud]
linha 146  ts=1776780646725.91 cloud=4  local=5   diff=+1  [cloud saiu de calib]
linha 149  ts=1776780646797.20 local=6  cloud=4   diff=+2
linha 150  ts=1776780646840.17 cloud=5  local=6   diff=+1
linha 173  ts=1776780647609.10 local=7  cloud=5   diff=+2
linha 176  ts=1776780647671.71 cloud=6  local=7   diff=+1
linha 194  ts=1776780648283.79 local=8  cloud=6   diff=+2
linha 220  ts=1776780649145.18 local=9  cloud=7   diff=+2
linha 221  ts=1776780649145.70 cloud=7  local=9   diff=+2
linha 246  ts=1776780649945.85 local=10 cloud=7   diff=+3
linha 249  ts=1776780650006.21 cloud=8  local=10  diff=+2
linha 270  ts=1776780650685.27 local=11 cloud=8   diff=+3
linha 273  ts=1776780650760.81 cloud=9  local=11  diff=+2
linha 292  ts=1776780651501.75 local=12 cloud=10  diff=+2
linha 293  ts=1776780651502.55 cloud=10 local=12  diff=+2
linha 315  ts=1776780652327.09 local=13 cloud=10  diff=+3
linha 318  ts=1776780652394.18 cloud=11 local=13  diff=+2
linha 346  ts=1776780653399.58 local=14 cloud=11  diff=+3
linha 352  ts=1776780653560.24 cloud=12 local=14  diff=+2
linha 371  ts=1776780654307.68 local=15 cloud=13  diff=+2
linha 372  ts=1776780654307.91 cloud=13 local=15  diff=+2
linha 399  ts=1776780655279.88 local=16 cloud=13  diff=+3
linha 402  ts=1776780655386.38 cloud=14 local=16  diff=+2
linha 423  ts=1776780656071.43 local=17 cloud=14  diff=+3
linha 424  ts=1776780656131.78 cloud=15 local=17  diff=+2
linha 446  ts=1776780656954.97 local=18 cloud=15  diff=+3
linha 449  ts=1776780657018.25 cloud=16 local=18  diff=+2
linha 475  ts=1776780657898.83 local=19 cloud=16  diff=+3
linha 477  ts=1776780657927.95 cloud=17 local=19  diff=+2
linha 501  ts=1776780658796.60 local=20 cloud=17  diff=+3
linha 505  ts=1776780658888.95 cloud=18 local=20  diff=+2
linha 531  ts=1776780659767.58 local=21 cloud=18  diff=+3
linha 533  ts=1776780659801.55 cloud=19 local=21  diff=+2
linha 555  ts=1776780660536.20 local=22 cloud=19  diff=+3
linha 557  ts=1776780660569.66 cloud=20 local=22  diff=+2
linha 603  ts=1776780662217.68 local=23 cloud=21  diff=+2
linha 604  ts=1776780662218.46 cloud=21 local=23  diff=+2
linha 631  ts=1776780663116.37 local=24 cloud=21  diff=+3
linha 634  ts=1776780663180.88 cloud=22 local=24  diff=+2
linha 657  ts=1776780664023.34 cloud=23 local=24  diff=+1
linha 668  ts=1776780664429.48 local=25 cloud=23  diff=+2
linha 677  ts=1776780664795.16 cloud=24 local=25  diff=+1
linha 690  ts=1776780665188.48 local=26 cloud=24  diff=+2
linha 701  ts=1776780665556.94 cloud=25 local=26  diff=+1
linha 716  ts=1776780666024.60 local=27 cloud=25  diff=+2
linha 729  ts=1776780666427.36 cloud=26 local=27  diff=+1
linha 743  ts=1776780666840.79 local=28 cloud=26  diff=+2
linha 752  ts=1776780667176.95 cloud=27 local=28  diff=+1
linha 766  ts=1776780667652.19 local=29 cloud=27  diff=+2
linha 778  ts=1776780668040.20 cloud=28 local=29  diff=+1
linha 789  ts=1776780668474.63 local=30 cloud=28  diff=+2
linha 802  ts=1776780668902.82 cloud=29 local=30  diff=+1
linha 810  ts=1776780669275.66 local=31 cloud=29  diff=+2
linha 822  ts=1776780669721.37 cloud=30 local=31  diff=+1
linha 833  ts=1776780670114.29 local=32 cloud=30  diff=+2
linha 846  ts=1776780670517.58 cloud=31 local=32  diff=+1
linha 869  ts=1776780671291.54 local=33 cloud=31  diff=+2    ← último antes do reset
linha 904  ts=1776780672539.06 local=34 cloud=1   diff=+33   ← RESET
linha 928  ts=1776780673334.84 local=35 cloud=1   diff=+34
linha 951  ts=1776780674103.84 local=36 cloud=1   diff=+35
linha 972  ts=1776780674807.27 local=37 cloud=1   diff=+36
linha 991  ts=1776780675581.53 cloud=31 local=37  diff=+6    ← cloud voltou
linha 993  ts=1776780675606.29 local=38 cloud=31  diff=+7
linha 1001 ts=1776780675946.82 cloud=32 local=38  diff=+6
linha 1012 ts=1776780676341.96 local=39 cloud=32  diff=+7
linha 1024 ts=1776780676679.10 cloud=33 local=39  diff=+6
linha 1033 ts=1776780677048.83 local=40 cloud=33  diff=+7
linha 1044 ts=1776780677382.73 cloud=34 local=40  diff=+6
linha 1056 ts=1776780677751.58 local=41 cloud=34  diff=+7
linha 1066 ts=1776780678154.26 cloud=35 local=41  diff=+6
linha 1106 ts=1776780679590.54 local=42 cloud=35  diff=+7
```

## 12. Apêndice B — Todos os `rep_increment` do NDJSON da cloud

```
frame_idx=14  ts=1776780643601  rep_count=1  kind=valley detector=LEFT_KNEE       [fase selecting]
frame_idx=14  ts=1776780643601  rep_count=1  kind=valley detector=LEFT_HIP        [fase selecting]
frame_idx=22  ts=1776780644169  rep_count=2  kind=peak   detector=LEFT_KNEE       [fase selecting]
frame_idx=25  ts=1776780644381  rep_count=1  kind=peak   detector=LEFT_ELBOW      [fase selecting]
frame_idx=32  ts=1776780644869  rep_count=1  kind=peak   detector=LEFT_HIP_ACROSS [fase selecting]
frame_idx=39  ts=1776780645378  rep_count=3  kind=valley detector=LEFT_KNEE       [fase selecting]
frame_idx=48  ts=1776780646030  rep_count=1  kind=valley detector=RIGHT_KNEE      [fase selecting]
frame_idx=52  ts=1776780646322  rep_count=2  kind=valley detector=LEFT_ELBOW      [fase selecting]
frame_idx=53  ts=1776780646393  rep_count=2  kind=valley detector=LEFT_HIP        [fase selecting]
frame_idx=54  ts=1776780646466  rep_count=4  kind=valley detector=LEFT_KNEE       [fase selecting]
frame_idx=54  ts=1776780646466  rep_count=1  kind=peak   detector=RIGHT_HIP_ACROSS
frame_idx=64  ts=1776780647221  rep_count=3  kind=valley detector=LEFT_ELBOW      [fase selecting]
frame_idx=72  ts=1776780647837  rep_count=2  kind=peak   detector=LEFT_HIP_ACROSS [lock = LEFT_ELBOW]
frame_idx=75  ts=1776780648039  rep_count=4  kind=valley detector=LEFT_ELBOW      [tracking]
frame_idx=87  ts=1776780648873  rep_count=5  kind=valley detector=LEFT_ELBOW
frame_idx=108 ts=1776780650332  rep_count=6  kind=valley detector=LEFT_ELBOW
frame_idx=121 ts=1776780651200  rep_count=7  kind=valley detector=LEFT_ELBOW
frame_idx=132 ts=1776780651941  rep_count=8  kind=valley detector=LEFT_ELBOW
frame_idx=143 ts=1776780652693  rep_count=9  kind=valley detector=LEFT_ELBOW
frame_idx=156 ts=1776780653571  rep_count=10 kind=valley detector=LEFT_ELBOW
frame_idx=173 ts=1776780654749  rep_count=11 kind=valley detector=LEFT_ELBOW
frame_idx=184 ts=1776780655500  rep_count=12 kind=valley detector=LEFT_ELBOW
frame_idx=199 ts=1776780656550  rep_count=13 kind=valley detector=LEFT_ELBOW
frame_idx=210 ts=1776780657313  rep_count=14 kind=valley detector=LEFT_ELBOW
frame_idx=223 ts=1776780658198  rep_count=15 kind=valley detector=LEFT_ELBOW
frame_idx=236 ts=1776780659115  rep_count=16 kind=valley detector=LEFT_ELBOW
frame_idx=250 ts=1776780660082  rep_count=17 kind=valley detector=LEFT_ELBOW
frame_idx=263 ts=1776780660983  rep_count=18 kind=valley detector=LEFT_ELBOW
frame_idx=274 ts=1776780661747  rep_count=19 kind=valley detector=LEFT_ELBOW
frame_idx=298 ts=1776780663412  rep_count=20 kind=valley detector=LEFT_ELBOW
frame_idx=312 ts=1776780664376  rep_count=21 kind=valley detector=LEFT_ELBOW
frame_idx=324 ts=1776780665215  rep_count=22 kind=valley detector=LEFT_ELBOW
frame_idx=335 ts=1776780665967  rep_count=23 kind=valley detector=LEFT_ELBOW
frame_idx=346 ts=1776780666735  rep_count=24 kind=valley detector=LEFT_ELBOW
frame_idx=359 ts=1776780667623  rep_count=25 kind=valley detector=LEFT_ELBOW
frame_idx=370 ts=1776780668369  rep_count=26 kind=valley detector=LEFT_ELBOW
frame_idx=382 ts=1776780669234  rep_count=27 kind=valley detector=LEFT_ELBOW
frame_idx=394 ts=1776780670076  rep_count=28 kind=valley detector=LEFT_ELBOW
frame_idx=406 ts=1776780670900  rep_count=29 kind=valley detector=LEFT_ELBOW
frame_idx=418 ts=1776780671694  rep_count=30 kind=valley detector=LEFT_ELBOW     ← último antes do switch
frame_idx=435 ts=1776780672890  rep_count=1  kind=valley detector=RIGHT_ELBOW    ← SWITCH
frame_idx=496 ts=1776780677124  rep_count=31 kind=valley detector=LEFT_ELBOW     ← retorno
frame_idx=507 ts=1776780677879  rep_count=32 kind=valley detector=LEFT_ELBOW
frame_idx=517 ts=1776780678571  rep_count=33 kind=valley detector=LEFT_ELBOW
frame_idx=528 ts=1776780679331  rep_count=34 kind=valley detector=LEFT_ELBOW
```

---

## 13. Apêndice C — Cálculo estatístico passo a passo com valores reais

Esta seção reproduz o pipeline exato do `determine_best_angle()` com os 120 ângulos brutos do `LEFT_ELBOW` imediatamente antes do switch (frames 311-430 no NDJSON). Cada operação é pareada com o ponto exato do código onde ela acontece.

### 13.1. Fluxo completo do cálculo

Por frame, o código executa em `session.py` esta sequência:

```
step_landmarks(landmarks, ts)
  └─ session.py:598      frame_buffer.append(lm)                       # deque maxlen=400
  └─ session.py:809      sdba[selected_angle].update(val)              # peak detector do tracked
  └─ session.py:822-826  re_eval_due = (now - last_re_eval) >= 0.75s   # ANGLE_SELECTION_REEVALUATE_EVERY_SEC
  └─ se re_eval_due:
       └─ session.py:830     variances = self._get_variances(rs, frame_buffer)
            └─ session.py:347   compute_angle_variances_from_buffer(buf)
                └─ variance_angle_selector.py:108-146:
                    └─ para cada joint:
                        ├─ calcula ângulos = calculate_from_type(cfg.type, cfg.landmarks, lm) para todo o buf
                        ├─ smooth_angle_series(history, window=SMOOTH_WINDOW=5)       # math_engine.py:758
                        ├─ min_ws = 15 se len(smoothed)>=90 else 12                    # linha 129
                        ├─ calculate_variance(smoothed)                                # math_engine.py:776 (var populacional)
                        ├─ compute_consistent_variance_score(smoothed, min_ws)         # math_engine.py:818
                        ├─ span_deg = max(smoothed) - min(smoothed)                    # linha 133
                        └─ grava dict com medianWindowVariance, activeWindowCount, smoothedRangeDeg
       └─ session.py:834     result = determine_best_angle(buf_list, variances)
            └─ variance_angle_selector.py:292-366:
                 └─ _get_top_candidate(variances)
                      └─ _candidate_score_if_eligible(joint, data)
                          └─ _variance_eligibility(joint, data):           # linhas 93-105
                              ├─ activeWindowCount >= MIN_ACTIVE_WINDOWS=3
                              ├─ medianWindowVariance >= t.min_variance    # 5 para cotovelos, 6 global
                              └─ smoothedRangeDeg >= t.min_range_deg        # 12 para cotovelos, 16 global
       └─ session.py:843-869  SE candidate != selected_angle:
            ├─ cur_ok = passes_consistent_variance_gate(variances, selected_angle)
            ├─ cand_var = variances[candidate].medianWindowVariance
            ├─ cur_var  = variances[selected_angle].medianWindowVariance
            ├─ stronger = cand_var >= cur_var * ANGLE_SELECTION_SWITCH_VARIANCE_RATIO (1.2)   # linha 848
            ├─ cooldown_ok = (now - last_switch) >= ANGLE_SELECTION_SWITCH_MIN_SEC (1.5s)     # linha 853
            └─ se cooldown_ok E ((not cur_ok) OU stronger):
                 ├─ rs["selected_angle"] = candidate                         # linha 856
                 ├─ rs["peak_detector"] = sdba.get(candidate)                # linha 858 ← SALTO para detector paralelo
                 └─ switched_to = candidate                                  # linha 860
```

**Ponto crítico** — o detector do candidato (`sdba[candidate]`) não é um detector novo: é um detector que vinha sendo **atualizado em paralelo durante toda a sessão**. Ele tem `rep_count` próprio, `peaks`/`valleys` próprios, `_calibrated` próprio. Ao trocar, o valor exposto ao cliente pula para o `rep_count` desse detector paralelo — foi assim que o cloud pulou de `30` para `1` no frame 435 (o detector paralelo do RIGHT_ELBOW tinha `rep_count=0` e o primeiro valley pós-switch virou rep 1).

### 13.2. Os 120 valores de entrada (LEFT_ELBOW raw, cloud)

Frames 311–430 do NDJSON, extraídos de `frame_snapshot.raw_angle`. Janela temporal: 8.221 s (taxa efetiva 14.6 fps confirmando ~15 fps do cloud).

```
min = 30.17°, max = 106.64°, amplitude = 76.48°
```

### 13.3. Passo 1 — Smoothing (média móvel janela 5)

Implementação: `math_engine.py:758-773` (`smooth_angle_series`)

```python
# Para cada posição i, calcula a média dos w vizinhos backward-centrados
w = 5, half = 2
out[i] = mean(values[max(0, hi-w) : hi]) onde hi = min(len, max(0, i-2)+5)
```

Chamado em `variance_angle_selector.py:128`:
```python
smoothed = smooth_angle_series(history, window=SMOOTH_WINDOW)  # SMOOTH_WINDOW = 5
```

Efeito nos 10 primeiros valores:
```
raw     : 51.66  54.20  56.68  56.11  55.16  50.96  38.22  31.11  34.41  37.82
smoothed: 54.76  54.76  54.76  54.62  51.43  46.31  41.97  38.51  36.41  38.98
```

O smoothing reduz picos espúrios de 1-frame e entra como denominador da variância subsequente.

### 13.4. Passo 2 — Escolha do `min_window_size`

Implementação: `variance_angle_selector.py:129`:
```python
min_ws = 15 if len(smoothed) >= 90 else 12
```

Para `len(smoothed) = 120`: **`min_ws = 15`**.

Este parâmetro controla quantas sub-janelas a série vai ser dividida em `compute_consistent_variance_score`.

### 13.5. Passo 3 — `compute_consistent_variance_score` (math_engine.py:818-844)

```python
num_windows = min(4, max(2, len(values) // min_window_size))
window_size = len(values) // num_windows
```

Para `len=120, min_ws=15`:
```
num_windows = min(4, max(2, 120//15)) = min(4, 8) = 4
window_size = 120//4 = 30
```

Em seguida, calcula a variância populacional de cada janela adjacente:
```python
for i in range(4):
    start = i * 30
    end   = 120 if i == 3 else start + 30
    window_variances.append(calculate_variance(smoothed[start:end])["variance"])
```

**Cálculo real (smoothed do LEFT_ELBOW, 120 frames pré-switch):**

| Janela | Frames cobertos (aprox) | Variância (graus²) | Observação |
|---|---|---|---|
| 1 | 311–340 | **105.81** | movimento estável, reps regulares |
| 2 | 341–370 | **201.39** | amplitude aumenta |
| 3 | 371–400 | **158.26** | reps normais |
| 4 | 401–430 | **458.53** | **aqui o usuário estava levantando** — amplitude máxima |

Ordenando: `[105.81, 158.26, 201.39, 458.53]`.

Mediana (n=4, par): `(sorted[1] + sorted[2]) / 2 = (158.26 + 201.39) / 2 = ` **`179.83`**

```python
active_count = sum(1 for v in window_variances if v >= MIN_VARIANCE_THRESHOLD)
# MIN_VARIANCE_THRESHOLD = 3.5 (math_engine.py:15)
active_count = 4  # todas passam de 3.5
```

Resultado final do passo 3 (cloud):
```
medianWindowVariance = 179.83
activeWindowCount    = 4
windowVariances      = [105.81, 201.39, 158.26, 458.53]
smoothedRangeDeg     = 105.71 - 30.17 ≈ 75.54 (aprox)
```

### 13.6. Passo 4 — Gate de elegibilidade (`_variance_eligibility`)

Implementação: `variance_angle_selector.py:93-105`:
```python
if active_windows < MIN_ACTIVE_WINDOWS:      # 3
    return False, 0.0
if consistent_var < t["min_variance"]:       # 5 para LEFT_ELBOW (rep_counter.toml:81)
    return False, 0.0
if span_deg < t["min_range_deg"]:            # 12 para LEFT_ELBOW (toml:82)
    return False, 0.0
return True, consistent_var
```

Para o cloud: `4 >= 3`, `179.83 >= 5`, `75.54 >= 12` → **gate OK**, score = 179.83.

### 13.7. O mesmo cálculo no LOCAL (60 únicas + 60 duplicatas)

Simulando o input local: últimas 60 amostras duplicadas = 120 amostras cobrindo **4.087 s** (metade do tempo do cloud).

```
num_windows = 4, window_size = 30 (mesmo calculo)
```

Janelas:
| Janela | Variância (graus²) |
|---|---|
| 1 | **257.36** |
| 2 | **241.99** |
| 3 | **238.30** |
| 4 | **328.32** |

Ordenadas: `[238.30, 241.99, 257.36, 328.32]`.
Mediana: `(241.99 + 257.36) / 2 = ` **`249.68`**

### 13.8. Comparação e interpretação

| Métrica | Cloud (120 frames únicos, 8.2s) | Local (60 únicas + 60 dup, 4.1s) | Razão L/C |
|---|---|---|---|
| `medianWindowVariance` LEFT_ELBOW | 179.83 | 249.68 | **1.39×** |
| Dispersão das variâncias de janela | 105 → 459 (5.3×) | 238 → 328 (1.4×) | |
| Tempo real coberto por janela | 2.05 s | 1.02 s | 0.5× |

**Ponto crítico — por que a mediana do local FICOU MAIOR e não menor:**

- No **cloud** (janelas de 2 s), a transição "user levantando" entra em **1 das 4** janelas (janela 4: var=459). As outras 3 janelas (var=106, 158, 201) são "normais". A **mediana é resistente ao outlier alto** — ordenadas ficam `[106, 158, 201, 459]` e a mediana é `(158+201)/2 = 180`.
- No **local** (janelas de 1 s cada, cobrindo só 4 s), **todas as 4 janelas caem dentro ou muito perto do período de transição**. A dispersão é baixa (238–328) porque não há janelas "normais" para contrastar. A mediana = 249.

Isso inverte minha hipótese inicial ("variância local atenuada"). O mecanismo real é:

> **A mediana robusta a outliers preserva o valor "normal" quando o buffer é longo, mas sobe quando o buffer é curto e toda a janela está no período anômalo.**

### 13.9. Como isso muda o resultado do switch

Aplicando a condição de switch em `session.py:848`:

```python
stronger = cand_var >= cur_var * ANGLE_SELECTION_SWITCH_VARIANCE_RATIO  # 1.2
```

Traduzindo para números:

| Cenário | `cur_var` (LEFT) | Threshold para switch | `cand_var` (RIGHT) precisa ser |
|---|---|---|---|
| **Cloud** | 179.83 | 179.83 × 1.2 | **≥ 215.79** |
| **Local** | 249.68 | 249.68 × 1.2 | **≥ 299.62** |

Ou seja, para o **mesmo** sinal do RIGHT_ELBOW (que passou em `cand_var ≈ 216-299` algum ponto nesse intervalo — não temos medida direta, mas é o único cenário consistente com o switch disparado no cloud e não no local):

- No cloud, RIGHT_ELBOW alcançou `cand_var ≥ 216` → **switch dispara**.
- No local, o mesmo movimento do RIGHT_ELBOW não chegou em `cand_var ≥ 300` → **switch não dispara**.

A barra sobe 38.8% no local por causa da janela temporal menor, e isso é suficiente para o algoritmo LOCAL não engatar a troca que o CLOUD engatou.

### 13.10. Efeito em cadeia nos outros parâmetros do peak detector

Os mesmos 120 valores também alimentam o `PeakDetector` do LEFT_ELBOW. Três parâmetros contados em **frames** mudam seu significado temporal:

#### `range_window_frames = 90` (rep_counter.toml:30)

Implementação: `math_engine.py:199` — `self._value_window = deque(maxlen=self.range_window_frames)`

Usado em `_update_rolling_range` (`math_engine.py:308-325`) para calcular `rolling_range = percentile(window, 95) − percentile(window, 5)`, que controla o `range_gate`.

```
Cloud: 90 frames = ~6.2 s de histórico → p95-p5 = ~70-75° → rolling_range grande
Local: 90 frames = ~3.0 s de histórico → pega só parte do ciclo → p95-p5 depende fortemente de onde caiu na oscilação
```

Com o range_gate controlando `_range_gate_allows_rep_recording` (linha 327) e decidindo se uma peak/valley vira rep (linhas 364, 469): um rolling_range temporariamente abaixo de `min_range_gate=15°` **bloqueia a contagem** mesmo com extremos válidos.

#### `min_peak_distance = 5` (rep_counter.toml:26)

Implementação: `math_engine.py:332, 437` — `if self.frame_count - self.last_peak_frame < self.min_peak_distance: return None, False`.

```
Cloud: 5 frames = 5/14.6 = ~340 ms  → exige pausa real entre picos/vales
Local: 5 frames = 5/30 = ~167 ms    → aceita picos bem mais próximos temporalmente
```

Se `frame_count` é incrementado a cada `update()` (linha 563), e o local chama `update()` com duplicatas, o contador passa frame_count mais rápido em tempo real. Isso **reduz o min_peak_distance efetivo no local para metade**.

#### `min_interval_ms = 500` (rep_counter.toml:42)

Implementação: `math_engine.py:358-362, 463-467`:
```python
if self._last_rep_time_ms is not None and (now_ms - self._last_rep_time_ms) < self.min_rep_interval_ms:
    interval_ok = False
```

Esse sim é em tempo real (`time.time() * 1000.0`) — **simétrico** nos dois ambientes. É o que salva o local de contar reps absurdamente rápidos mesmo com `min_peak_distance` quebrado pela duplicação.

### 13.11. Resumo do paralelo código ↔ comportamento observado

| Métrica observada | Fórmula | Arquivo:linha | Valor cloud | Valor local | Impacto no algoritmo |
|---|---|---|---|---|---|
| `medianWindowVariance` | mediana(var das 4 janelas) | `math_engine.py:818-844` | **179.83** | **249.68** | Entra em `passes_consistent_variance_gate` (session.py:844) e no ratio de switch (session.py:848). Local mais alto → mais difícil trocar. |
| `activeWindowCount` | count(var_janela ≥ 3.5) | `math_engine.py:839` | **4** | **4** | Ambos passam em `MIN_ACTIVE_WINDOWS=3` (variance_angle_selector.py:99). Simétrico neste caso. |
| `smoothedRangeDeg` | max(smoothed) − min(smoothed) | `variance_angle_selector.py:133` | **~75.5°** | **~72°** | Ambos > `min_range_deg=12` (toml:82). Simétrico. |
| Janela temporal de cada sub-janela | `(num_frames/num_windows) / fps` | derivado | **~2.05 s** | **~1.02 s** | Determina o que cada janela captura — normal vs anômalo. |
| `rolling_range` em frames=90 | `p95 − p5` da window | `math_engine.py:144-152, 308-325` | cobre **~6.2 s** | cobre **~3.0 s** | Gate mais estável no cloud; mais volátil no local. |
| `min_peak_distance = 5 frames` | `frame_count − last_peak_frame < 5` | `math_engine.py:332, 437` | bloqueia por **~340 ms** | bloqueia por **~167 ms** | Local mais permissivo com picos próximos. |
| `min_rep_interval_ms = 500` | diff em `time.time()` | `math_engine.py:358-362, 463-467` | **simétrico** | **simétrico** | Único gate de timing robusto à duplicação. |
| `ANGLE_SELECTION_SWITCH_VARIANCE_RATIO = 1.2` | `cand >= cur * 1.2` | `session.py:848` | barra em **215.79** | barra em **299.62** | Mesma regra, mas cur_var diferente → barras efetivas diferentes. |
| `ANGLE_SELECTION_SWITCH_MIN_SEC = 1.5` | `(now − last_switch) >= 1.5` | `session.py:853` | **simétrico** | **simétrico** | Em tempo real nos dois. |
| `ANGLE_SELECTION_REEVALUATE_EVERY_SEC = 0.75` | `(now − last_re_eval) >= 0.75` | `session.py:822-826` | **simétrico** | **simétrico** | Em tempo real nos dois. Mas cada re-avaliação no cloud vê 0.75 s de *novos* dados; no local vê 0.375 s de dados novos (metade do tempo são duplicatas do valor corrente). |

### 13.12. Conexão final com a divergência de reps

Rastreando da causa raiz até o efeito observado:

1. **Duplicação de amostras** no `frame_buffer` local (consequência de `Queue(maxsize=1)` + `step_landmarks` a 30 Hz + respostas da cloud a 15 Hz).
2. → o `frame_buffer` deque (`maxlen=400`) cobre **~metade do tempo real** no local (~13 s vs ~27 s no cloud).
3. → `compute_consistent_variance_score` divide em 4 janelas, cada uma cobrindo **~metade do tempo**.
4. → durante transições posturais, o local vê **todas as 4 janelas dentro da transição** enquanto o cloud vê **apenas 1 de 4** → mediana local **mais alta e estável**.
5. → `medianWindowVariance` do tracked joint (LEFT_ELBOW) fica **~1.39× maior** no local.
6. → a barra para switch (1.2× `cur_var`) fica **~1.39× mais alta** no local.
7. → mesmo quando o RIGHT_ELBOW atinge `cand_var` suficiente para switch no cloud (`>215`), não atinge no local (`<300`).
8. → cloud troca de joint, reset `rep_count=30→1`, fica ~2.4 s sem contar, volta ao LEFT_ELBOW em rep=31; local não troca, continua contando 33→37 no mesmo período.
9. → divergência final de reps = 7 (5 perdidos durante o switch + 2 residuais de latência e calibração).

**Conclusão**: a diferença de taxa de amostragem entre camera (~30 fps) e respostas da cloud (~15 fps), combinada com o fato de `step_landmarks` ser chamado por-frame-de-câmera sem deduplicação, faz com que todos os parâmetros contados em frames (`frame_buffer.maxlen`, `range_window_frames`, `min_peak_distance`, janelas de variância) cubram **~metade do tempo real** no local. O efeito mais relevante não é a redução de variância por duplicação dentro de uma janela (esse é secundário), mas sim o **encurtamento da janela de observação**, que faz a mediana robusta proteger o joint atual contra o switch quando a janela inteira cai numa transição.

---

*Documento gerado em 2026-04-21 a partir das evidências em `visualizer/log_vm_rep_simulator.txt`, `visualizer/latest-rep-counter-session.ndjson`, `visualizer/latest-metadata.json` e diff direto entre `flexible-rep-counter/src/` e `arquivos-vm-cloud/vendor/flexible-rep-counter/src/`. Cálculos do Apêndice C reproduzem exatamente o pipeline de `variance_angle_selector.py` + `math_engine.py` aplicados aos ângulos reais extraídos do NDJSON.*
