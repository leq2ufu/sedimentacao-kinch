"""
Sedimentação — Método de Kynch
Curva clássica: altura da interface z (cm) vs. tempo t (s)

Detecção: contagem de pixels turbulentos na escala de cinza [gray_min, gray_max].
Correção: ajusta apenas os pontos iniciais ruins (antes da região linear estável),
preservando integralmente os pontos da zona não-linear/compressão.
"""

import io
import os
import shutil
import tempfile

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from scipy.signal import savgol_filter

st.set_page_config(page_title="Sedimentação Kinch", layout="wide",
                   initial_sidebar_state="expanded")

for k, v in [("results", None), ("video_path", None), ("video_hash", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Funções ────────────────────────────────────────────────────────────────────

def get_auto_crop(image_bgr):
    non_black = np.any(image_bgr != 0, axis=2)
    cols = np.where(non_black.any(axis=0))[0]
    if len(cols) == 0:
        return 0, image_bgr.shape[1]
    left, right = cols[0], cols[-1]
    w = right - left
    return int(left + 0.40 * w), int(left + 0.60 * w)


def calcular_altura_raw(frame_bgr, crop_left, crop_right, y1, y2, gray_min, gray_max):
    roi  = frame_bgr[y1:y2, crop_left:crop_right]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = (gray >= gray_min) & (gray <= gray_max)
    frac = float(np.sum(mask)) / max(gray.size, 1)
    return frac


def normalizar_alturas(fracoes, z0):
    if not fracoes or fracoes[0] == 0:
        return [0.0] * len(fracoes)
    base = fracoes[0]
    return [z0 * f / base for f in fracoes]


def _melhor_reta(t, h, n):
    """Busca a janela [a, b) de maior R² para a fase de velocidade constante."""
    a_max = max(3, int(n * 0.45))      # início da reta cabe nos primeiros 45%
    b_min = max(8, int(n * 0.12))      # reta tem ao menos ~12% dos pontos
    melhor = dict(r2=-np.inf, a=0, b=min(n, 10), slope=0.0, intercept=0.0)

    for a in range(0, a_max):
        for b in range(a + b_min, n + 1):
            slope, intercept, r, *_ = stats.linregress(t[a:b], h[a:b])
            if r * r > melhor["r2"]:
                melhor = dict(r2=r * r, a=a, b=b, slope=slope, intercept=intercept)
    return melhor


def corrigir_regiao_linear(tempos, alturas):
    """
    Produz uma curva limpa, sem dentes, generalista para qualquer ensaio:

    1. Acha a janela de maior linearidade (fase de velocidade constante).
    2. Substitui TODA essa janela pelos valores EXATOS da reta ajustada
       (a região de velocidade constante fica perfeitamente reta).
    3. A zona de compressão (após a reta) é suavizada com Savitzky-Golay.
    4. Impõe monotonicidade não-crescente em toda a curva (a interface só
       pode descer): isso elimina QUALQUER dente/sobressalto, em qualquer
       posição, pois um dente é fisicamente impossível.
    """
    n = len(tempos)
    t = np.array(tempos, dtype=float)
    h = np.array(alturas, dtype=float)

    melhor   = _melhor_reta(t, h, n)
    a, b     = melhor["a"], melhor["b"]
    slope    = melhor["slope"]
    intercept = melhor["intercept"]
    r2       = melhor["r2"]

    h_line = slope * t + intercept
    h_corr = h.copy()

    # (2) região de velocidade constante = reta exata
    h_corr[a:b] = h_line[a:b]
    # pontos iniciais ruins (antes de 'a') também projetados na reta
    h_corr[:a] = h_line[:a]

    # (3) suaviza a zona de compressão
    tail = h_corr[b:]
    if len(tail) >= 7:
        win = min(len(tail) if len(tail) % 2 == 1 else len(tail) - 1, 11)
        if win >= 5:
            h_corr[b:] = savgol_filter(tail, win, 2)

    # (4) monotonicidade não-crescente → remove todos os dentes
    h_corr = np.minimum.accumulate(h_corr)

    return h_corr.tolist(), slope, intercept, r2, (a, b)


# ── Barra lateral ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configurações")

    video_file = st.file_uploader(
        "Carregar vídeo", type=["mp4", "avi", "mov", "mkv", "wmv"]
    )

    st.divider()
    st.subheader("Recorte Vertical")
    corte_topo = st.slider("Corte no topo (% da imagem)", 0, 20, 2)
    corte_base = st.slider("Corte na base (% da imagem)", 0, 20, 2)

    st.divider()
    st.subheader("Recorte Horizontal")
    auto_crop = st.checkbox("Detecção automática da proveta", value=True)
    if not auto_crop:
        col_center = st.slider("Centro horizontal (px)", 0, 1920, 540)
        col_half   = st.slider("Meia-largura (px)", 5, 200, 35)
    else:
        col_center = col_half = None

    st.divider()
    st.subheader("Limiar da Escala de Cinza")
    gray_min = st.slider("Cinza mínimo (zona turva)", 0, 200, 92)
    gray_max = st.slider("Cinza máximo (zona turva)", 100, 255, 240)

    st.divider()
    st.subheader("Amostragem")
    z0             = st.number_input("z₀ — altura inicial (cm)", 1.0, 200.0, 34.0, step=0.5)
    frame_step_s   = st.slider("Intervalo entre capturas (s)", 1, 120, 8)
    frames_ignorar = st.slider("Frames iniciais a ignorar", 0, 50, 15)

    st.divider()
    auto_correct = st.checkbox("Corrigir região linear inicial", True)

# ── Cabeçalho ─────────────────────────────────────────────────────────────────
st.title("🧪 Sedimentação — Método de Kynch")
st.caption("Curva clássica · z (cm) vs. t (s)")

if video_file is None:
    st.info("👈 Carregue um vídeo na barra lateral para começar.")
    st.stop()

# ── Salvar vídeo ──────────────────────────────────────────────────────────────
# Identifica o arquivo por nome+tamanho (sem ler tudo na RAM) e grava em disco
# em blocos, evitando segurar uma cópia inteira do vídeo na memória.
file_hash = hash((video_file.name, video_file.size))

if file_hash != st.session_state.video_hash:
    st.session_state.results    = None
    st.session_state.video_hash = file_hash
    old = st.session_state.video_path
    if old and os.path.exists(old):
        try:
            os.unlink(old)
        except Exception:
            pass
    video_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        shutil.copyfileobj(video_file, f, length=4 * 1024 * 1024)  # blocos de 4 MB
        st.session_state.video_path = f.name

video_path = st.session_state.video_path

cap          = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
h_vid        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w_vid        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ret, frame0  = cap.read()
cap.release()

st.success(
    f"**{w_vid}×{h_vid} px** &nbsp;·&nbsp; **{fps:.2f} FPS** &nbsp;·&nbsp; "
    f"**{total_frames / fps:.1f} s** &nbsp;({total_frames} frames)"
)

# Crop coords
if auto_crop and ret:
    crop_left, crop_right = get_auto_crop(frame0)
else:
    crop_left  = max(0, (col_center or w_vid // 2) - (col_half or 35))
    crop_right = min(w_vid, (col_center or w_vid // 2) + (col_half or 35))

y1 = max(0, int(h_vid * corte_topo / 100))
y2 = min(h_vid, h_vid - int(h_vid * corte_base / 100))

st.caption(f"Corte horizontal: coluna {crop_left}–{crop_right} px  |  "
           f"Corte vertical: linha {y1}–{y2} px")

st.divider()

# ── Processamento ─────────────────────────────────────────────────────────────
if st.button("🚀 Iniciar Processamento", type="primary", use_container_width=True):
    st.session_state.results = None

    frame_step = max(1, int(frame_step_s * fps))
    frame_idxs = list(range(0, total_frames, frame_step))
    n_steps    = len(frame_idxs)

    fracoes_raw, tempos_raw = [], []

    prog = st.progress(0.0)
    stat = st.empty()
    plot_live = st.empty()

    cap_proc = cv2.VideoCapture(video_path)

    # Leitura SEQUENCIAL com grab(): pula frames sem decodificar (rápido) e só
    # decodifica (retrieve) os frames amostrados. Evita o seek caro do cap.set().
    target_set = set(frame_idxs)
    next_target_i = 0
    cur_frame = 0
    # Atualiza o gráfico ao vivo no máximo ~25 vezes no total (leve)
    plot_every = max(1, n_steps // 25)

    while next_target_i < n_steps:
        if not cap_proc.grab():
            break
        if cur_frame in target_set:
            ret_f, frame = cap_proc.retrieve()
            if not ret_f:
                break
            t = cur_frame / fps
            try:
                frac = calcular_altura_raw(frame, crop_left, crop_right, y1, y2, gray_min, gray_max)
                fracoes_raw.append(frac)
                tempos_raw.append(t)
            except Exception:
                pass

            step = next_target_i
            # Curva ao vivo (esparsa, para não pesar)
            if (step % plot_every == 0 or step == n_steps - 1) and len(fracoes_raw) > frames_ignorar + 2:
                fracs_v = fracoes_raw[frames_ignorar:]
                t_v     = [tt - tempos_raw[frames_ignorar] for tt in tempos_raw[frames_ignorar:]]
                alts_v  = normalizar_alturas(fracs_v, float(z0))
                fig_lv  = go.Figure(go.Scatter(
                    x=t_v, y=alts_v, mode="lines",
                    line=dict(color="royalblue", width=2),
                ))
                fig_lv.update_layout(
                    xaxis_title="Tempo (s)", yaxis_title="z (cm)",
                    xaxis=dict(range=[0, max(t_v)]),
                    yaxis=dict(range=[0, float(z0) * 1.10]),
                    height=400, margin=dict(l=60, r=20, t=20, b=50),
                )
                plot_live.plotly_chart(fig_lv, use_container_width=True, key=f"lv_{step}")

            prog.progress((step + 1) / n_steps)
            stat.text(f"Processando: {step+1}/{n_steps} frames · t = {t:.1f} s")
            next_target_i += 1

        cur_frame += 1

    cap_proc.release()

    # Descartar frames iniciais
    fracoes_use = fracoes_raw[frames_ignorar:] if len(fracoes_raw) > frames_ignorar else fracoes_raw
    tempos_use  = tempos_raw [frames_ignorar:] if len(tempos_raw)  > frames_ignorar else tempos_raw

    if len(fracoes_use) < 5:
        st.error("Poucos pontos válidos. Ajuste os limiares de cinza ou o recorte.")
        st.stop()

    alturas = normalizar_alturas(fracoes_use, float(z0))
    t0      = tempos_use[0]
    tempos  = [t - t0 for t in tempos_use]

    # Correção linear
    alturas_corr = alturas[:]
    slope = intercept = r2 = 0.0
    intervalo = (0, 0)

    if auto_correct:
        alturas_corr, slope, intercept, r2, intervalo = corrigir_regiao_linear(tempos, alturas)

    st.session_state.results = dict(
        tempos=tempos, alturas=alturas, alturas_corr=alturas_corr,
        slope=slope, intercept=intercept, r2=r2, intervalo=intervalo,
        z0=float(z0), auto_correct=auto_correct,
    )

# ── Resultado final ────────────────────────────────────────────────────────────
res = st.session_state.results
if res:
    tempos      = res["tempos"]
    alturas_cor = res["alturas_corr"]
    slope       = res["slope"]
    intercept   = res["intercept"]
    r2          = res["r2"]
    intervalo   = res["intervalo"]
    z0_res      = res["z0"]

    if res["auto_correct"]:
        a, b = intervalo
        st.info(
            f"Região linear detectada: pontos **{a}–{b}** "
            f"(t = {tempos[a]:.0f}–{tempos[min(b, len(tempos)-1)]:.0f} s) · "
            f"R² = **{r2:.4f}** · "
            f"velocidade = **{abs(slope)*60:.4f} cm/min** · "
            f"zona de velocidade constante linearizada · compressão suavizada · "
            f"curva monotônica (sem dentes)"
        )

    st.subheader("Curva de Sedimentação — Resultado Final")

    fig_out = go.Figure()

    fig_out.add_trace(go.Scatter(
        x=tempos, y=alturas_cor,
        mode="lines",
        line=dict(color="royalblue", width=3),
        showlegend=False,
    ))

    t_max = max(tempos) * 1.02
    fig_out.update_layout(
        xaxis_title="Tempo (s)",
        yaxis_title="Altura da Interface z (cm)",
        xaxis=dict(range=[0, t_max]),
        yaxis=dict(range=[0, z0_res * 1.08]),
        height=560,
        showlegend=False,
        margin=dict(l=65, r=20, t=20, b=65),
    )
    st.plotly_chart(fig_out, use_container_width=True)

    # Métricas
    m1, m2, m3 = st.columns(3)
    m1.metric("Pontos processados", len(tempos))
    m2.metric("R² (região linear)", f"{r2:.4f}" if res["auto_correct"] else "—")
    m3.metric("Velocidade inicial", f"{abs(slope)*60:.4f} cm/min" if res["auto_correct"] else "—")

    # Excel
    df_out = pd.DataFrame({
        "Tempo (s)":   np.round(tempos, 2),
        "Tempo (min)": np.round(np.array(tempos) / 60, 3),
        "z (cm)":      np.round(alturas_cor, 3),
    })

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="Sedimentação Kinch")
        ws = writer.sheets["Sedimentação Kinch"]
        for col_ltr, width in zip("ABC", [14, 12, 12]):
            ws.column_dimensions[col_ltr].width = width
        from openpyxl.styles import Font
        for cell in ws[1]:
            cell.font = Font(bold=True)
    buf.seek(0)

    st.download_button(
        "📥 Baixar — Excel (.xlsx)",
        data=buf,
        file_name="curva_sedimentacao_kinch.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.spec",
        use_container_width=True,
    )
