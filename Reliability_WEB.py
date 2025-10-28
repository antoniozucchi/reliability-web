import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Tenta usar a biblioteca reliability; se não disponível, usa scipy
USE_RELIABILITY = True
try:
    from reliability.Fitters import Fit_Weibull_2P, Fit_Lognormal_2P, Fit_Exponential_1P
except Exception:
    USE_RELIABILITY = False
    from scipy.stats import weibull_min, lognorm, expon

st.set_page_config(page_title="Confiabilidade – Tempos até a Falha", layout="centered")
st.title("📊 Confiabilidade – Tempos até a Falha")
st.caption("Envie um CSV com **uma coluna** contendo os tempos até a falha (horas, ciclos, km, etc.).")

uploaded = st.file_uploader("📁 Escolha o CSV", type=["csv"])

with st.expander("Formato do CSV (exemplo)"):
    st.code("tempo\n120\n340\n87\n410\n...", language="text")
    st.write("• O app usa a **primeira coluna numérica**.\n• Use vírgula como separador (ou marque abaixo para ponto e vírgula).")

sep_semicolon = st.checkbox("Meu CSV usa ponto e vírgula ( ; )", value=False)

if uploaded:
    # Lê o CSV
    raw = uploaded.read()
    df = pd.read_csv(io.BytesIO(raw), sep=";" if sep_semicolon else ",")
    col = df.select_dtypes(include=[np.number]).columns
    if len(col) == 0:
        df[df.columns[0]] = pd.to_numeric(df[df.columns[0]], errors="coerce")
        col = [df.columns[0]]
    times = df[col[0]].dropna().values
    times = times[times > 0]

    if len(times) < 3:
        st.error("Precisamos de pelo menos 3 valores válidos.")
        st.stop()

    st.subheader("✅ Estatísticas básicas")
    s = pd.Series(times, name="tempos")
    st.write(pd.DataFrame({
        "n": [s.count()],
        "mín": [s.min()],
        "máx": [s.max()],
        "média": [s.mean()],
        "mediana": [s.median()],
        "desvio-padrão": [s.std(ddof=1)]
    }))

    # ---------- Ajuste de distribuições ----------
    if USE_RELIABILITY:
        wb = Fit_Weibull_2P(failures=times, show_plot=False)
        ln = Fit_Lognormal_2P(failures=times, show_plot=False)
        ex = Fit_Exponential_1P(failures=times, show_plot=False)
    else:
        c, loc, scale = weibull_min.fit(times, floc=0)
        s_ln, loc_ln, scale_ln = lognorm.fit(times, floc=0)
        loc_ex, scale_ex = 0, expon.fit(times, floc=0)[1]

    # ---------- Gráficos ----------
    st.subheader("📈 Gráficos de Confiabilidade")

    # 1️⃣ Histograma
    fig1, ax1 = plt.subplots()
    ax1.hist(times, bins="auto", density=True, color="lightblue", edgecolor="black")
    ax1.set_xlabel("Tempo até a falha")
    ax1.set_ylabel("Densidade")
    ax1.set_title("Histograma dos Tempos até a Falha")
    st.pyplot(fig1)

    # Função de sobrevivência empírica
    def empirical_survival(x):
        x = np.sort(x)
        n = len(x)
        ranks = np.arange(1, n + 1)
        sf = 1 - ranks / (n + 1)
        return x, sf

    grid = np.linspace(times.min(), times.max(), 200)

    # 2️⃣ Confiabilidade R(t)
    fig2, ax2 = plt.subplots()
    x_emp, R_emp = empirical_survival(times)
    ax2.step(x_emp, R_emp, where="post", label="Empírico", color="black")

    if USE_RELIABILITY:
        plt.sca(ax2)
        wb.distribution.SF(label="Weibull (ajuste)")
        plt.sca(ax2)
        ln.distribution.SF(label="Lognormal (ajuste)")
        plt.sca(ax2)
        ex.distribution.SF(label="Exponencial (ajuste)")
    else:
        ax2.plot(grid, weibull_min.sf(grid, c, loc=loc, scale=scale), label="Weibull")
        ax2.plot(grid, lognorm.sf(grid, s_ln, loc=loc_ln, scale=scale_ln), label="Lognormal")
        ax2.plot(grid, expon.sf(grid, loc=loc_ex, scale=scale_ex), label="Exponencial")

    ax2.set_xlabel("t")
    ax2.set_ylabel("R(t)")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.set_title("Função de Confiabilidade R(t)")
    st.pyplot(fig2)

    # 3️⃣ CDF (F(t))
    fig3, ax3 = plt.subplots()
    x_sorted = np.sort(times)
    F_emp = np.arange(1, len(times) + 1) / (len(times) + 1)
    ax3.step(x_sorted, F_emp, where="post", label="Empírico", color="black")

    if USE_RELIABILITY:
        plt.sca(ax3)
        wb.distribution.CDF(label="Weibull (ajuste)")
        plt.sca(ax3)
        ln.distribution.CDF(label="Lognormal (ajuste)")
        plt.sca(ax3)
        ex.distribution.CDF(label="Exponencial (ajuste)")
    else:
        ax3.plot(grid, weibull_min.cdf(grid, c, loc=loc, scale=scale), label="Weibull")
        ax3.plot(grid, lognorm.cdf(grid, s_ln, loc=loc_ln, scale=scale_ln), label="Lognormal")
        ax3.plot(grid, expon.cdf(grid, loc=loc_ex, scale=scale_ex), label="Exponencial")

    ax3.set_xlabel("t")
    ax3.set_ylabel("F(t)")
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.set_title("Função de Distribuição Acumulada F(t)")
    st.pyplot(fig3)

    # 4️⃣ Taxa de falha λ(t) (calculada manualmente)
    fig4, ax4 = plt.subplots()
    if USE_RELIABILITY:
        # Calcula manualmente usando PDF/SF
        t = grid
        f_w = wb.distribution.PDF(t)
        R_w = wb.distribution.SF(t)
        f_l = ln.distribution.PDF(t)
        R_l = ln.distribution.SF(t)
        f_e = ex.distribution.PDF(t)
        R_e = ex.distribution.SF(t)
        ax4.plot(t, f_w / np.maximum(R_w, 1e-12), label="Weibull")
        ax4.plot(t, f_l / np.maximum(R_l, 1e-12), label="Lognormal")
        ax4.plot(t, f_e / np.maximum(R_e, 1e-12), label="Exponencial")
    else:
        S_w = weibull_min.sf(grid, c, loc=loc, scale=scale)
        f_w = weibull_min.pdf(grid, c, loc=loc, scale=scale)
        S_l = lognorm.sf(grid, s_ln, loc=loc_ln, scale=scale_ln)
        f_l = lognorm.pdf(grid, s_ln, loc=loc_ln, scale=scale_ln)
        S_e = expon.sf(grid, loc=loc_ex, scale=scale_ex)
        f_e = expon.pdf(grid, loc=loc_ex, scale=scale_ex)
        ax4.plot(grid, f_w / np.maximum(S_w, 1e-12), label="Weibull")
        ax4.plot(grid, f_l / np.maximum(S_l, 1e-12), label="Lognormal")
        ax4.plot(grid, f_e / np.maximum(S_e, 1e-12), label="Exponencial")

    ax4.set_xlabel("t")
    ax4.set_ylabel("λ(t)")
    ax4.legend()
    ax4.set_title("Taxa de Falha λ(t)")
    st.pyplot(fig4)

    st.subheader("📋 Parâmetros estimados")
    if USE_RELIABILITY:
        # Compatibilidade: tenta acessar 'eta' e, se não existir, usa 'alpha'
        eta_value = getattr(wb, "eta", getattr(wb, "alpha", None))
        st.write({
            "Weibull": {"β (forma)": wb.beta, "η (escala)": eta_value},
            "Lognormal": {"μ": getattr(ln, "mu", None), "σ": getattr(ln, "sigma", None)},
            "Exponencial": {"λ": 1.0 / ex.Lambda if hasattr(ex, 'Lambda') else 'n/d'}
        })
    else:
        st.write("Usando ajuste via SciPy (Weibull, Lognormal, Exponencial).")
else:
    st.info("Envie um arquivo CSV para gerar os gráficos.")
