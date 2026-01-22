import streamlit as st
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import io

# Sideoppsett
st.set_page_config(page_title="Proff Matte-Analyse", layout="wide")
st.title("📊 Komplett Funksjonsanalyse & Fortegnsskjema")

# --- SIDEBAR: KONTROLLPANEL ---
with st.sidebar:
    st.header("1. Definer funksjon")
    input_f = st.text_input("Skriv inn f(x):", "(x - 3) / (x + 1)")
    
    with st.expander("ℹ️ Inntastingsguide"):
        st.markdown("""
        - **Potens:** `x**2`
        - **Gange:** `2*x`
        - **Brøk:** `(x+1)/(x-1)`
        - **Kvadratrot:** `sqrt(x)`
        """)

    st.header("2. Analyse-nivå")
    nivå = st.radio("Hva skal drøftes?", ["f(x)", "f'(x)", "f''(x)"])
    
    st.header("3. Lærer-modus")
    vis_konst = st.checkbox("Vis konstante tall", value=True)
    skjul_var = st.checkbox("Skjul faktorer med x", value=False)
    v_fasit = not st.checkbox("Skjul fasit og tekst", value=False)
    v_x_akse = not st.checkbox("Skjul x-verdier", value=False)
    
    st.header("4. Graf & Design")
    v_graf = st.checkbox("Vis grafen til f(x)", value=True)
    v_zoom = st.slider("Vertikal zoom (graf)", 1.0, 10.0, 2.0)
    farge_valg = st.selectbox("Fargetema", ["Svart", "Blå/Rød"])

# --- MATEMATISK LOGIKK ---
try:
    x = sp.symbols('x')
    f_orig = sp.sympify(input_f)
    
    # Forenkler de deriverte
    f1 = sp.simplify(sp.diff(f_orig, x))
    f2 = sp.simplify(sp.diff(f1, x))
    
    if nivå == "f(x)": target, label, grad = f_orig, "f(x)", 0
    elif nivå == "f'(x)": target, label, grad = f1, "f'(x)", 1
    else: target, label, grad = f2, "f''(x)", 2

    # Rydd opp i uttrykket
    target_clean = sp.cancel(target)
    teller, nevner = sp.fraction(target_clean)
    
    # Finn ALLE reelle røtter for x-aksen
    nullpunkter = sorted([p for p in sp.solve(teller, x) if p.is_real], key=float)
    bruddpunkter = sorted([p for p in sp.solve(nevner, x) if p.is_real], key=float)
    kritiske = sorted(list(set(nullpunkter + bruddpunkter)), key=float)

    # Bygg opp faktorer
    rader = []
    t_data = sp.factor_list(teller)
    n_data = sp.factor_list(nevner)
    
    # 1. Konstantledd
    konstant = t_data[0] / n_data[0]
    if abs(konstant - 1) > 1e-9:
        rader.append(('k', konstant))
    
    # 2. Variabelledd - HER SPLITTER VI OPP POTENSER
    # Vi går gjennom faktorene i teller og nevner
    for fkt, eksp in t_data[1] + n_data[1]:
        # Hvis eksponenten er f.eks. 2, legger vi til faktoren 2 ganger
        for _ in range(eksp):
            rader.append(('v', fkt))

    # 3. Totalen nederst
    rader.append(('t', target_clean))

    # --- TEGNEFUNKSJON SKJEMA ---
    def tegn_skjema():
        margin = 2.0
        x_min = float(kritiske[0]) - margin if kritiske else -5
        x_max = float(kritiske[-1]) + margin if kritiske else 5
        pts = sorted(list(set([x_min, x_max] + [float(v) for v in kritiske])))
        
        # Beregn antall linjer for å sette høyde
        fig, ax = plt.subplots(figsize=(12, len(rader) * 1.0))
        ax.xaxis.set_ticks_position('top')

        for idx, (type, uttr) in enumerate(reversed(rader)):
            y = idx
            # Navngiving
            if type == 't': n = label
            elif type == 'k': n = f"${sp.latex(uttr)}$" if vis_konst else "Konstant"
            else: n = f"Faktor {len(rader)-idx}" if skjul_var else f"${sp.latex(uttr)}$"
            
            ax.text(x_min - 0.2, y, n, va='center', ha='right', fontsize=13)
            
            # Tegn linjer
            for i in range(len(pts)-1):
                m = (pts[i] + pts[i+1]) / 2
                res = uttr.subs(x, m) if hasattr(uttr, 'subs') else uttr
                pos = res > 0
                c = 'black'
                if farge_valg == "Blå/Rød": c = 'blue' if pos else 'red'
                ax.plot([pts[i], pts[i+1]], [y, y], '-' if pos else '--', color=c, lw=2.5)

                # Piler og krumning på bunnlinjen
                if type == 't' and v_fasit:
                    if grad == 1: ax.text(m, y-0.35, r"$\nearrow$" if pos else r"$\searrow$", ha='center', fontsize=20, color='gray')
                    elif grad == 2: ax.text(m, y-0.35, r"$\cup$" if pos else r"$\cap$", ha='center', fontsize=20, color='gray')

            # Vertikale linjer og 0/X
            for p in kritiske:
                p_v = float(p)
                ax.axvline(p_v, color='gray', lw=0.6, linestyle=':', alpha=0.5)
                if type == 't':
                    is_b = any(sp.simplify(p-b)==0 for b in bruddpunkter)
                    ax.text(p_v, y, 'X' if is_b else '0', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none'))
                elif type == 'v':
                    try:
                        if abs(float(uttr.subs(x, p))) < 1e-9:
                            ax.text(p_v, y, '0', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none'))
                    except: pass

        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(-0.8, len(rader) + 0.5)
        
        if v_x_akse:
            ax.set_xticks([float(v) for v in kritiske])
            ax.set_xticklabels([f"${sp.latex(v)}$" for v in kritiske], fontsize=12)
        else: ax.set_xticks([])
        
        ax.spines['top'].set_visible(True)
        ax.spines[['bottom', 'left', 'right']].set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.tight_layout()
        return fig

    # --- VISNING ---
    if v_fasit:
        st.latex(f"{label} = {sp.latex(target_clean)}")
    
    st.pyplot(tegn_skjema())

    # --- GRAF ---
    if v_graf:
        st.divider()
        st.subheader("Grafisk fremstilling av f(x)")
        x_start = float(min(kritiske)) - 4 if kritiske else -7
        x_slutt = float(max(kritiske)) + 4 if kritiske else 7
        
        stasjonære = [p for p in sp.solve(f1, x) if p.is_real]
        y_skala = [float(f_orig.subs(x, p)) for p in stasjonære if np.isfinite(float(f_orig.subs(x, p)))]
        if not y_skala: y_skala = [-2, 2]
        y_mid, y_r = (max(y_skala)+min(y_skala))/2, (max(y_skala)-min(y_skala))
        if y_r < 0.1: y_r = 5.0

        fig_g, ax_g = plt.subplots(figsize=(10, 5))
        f_num = sp.lambdify(x, f_orig, "numpy")
        
        b_vals = sorted([float(p) for p in bruddpunkter])
        intervaller = [x_start] + b_vals + [x_slutt]
        for i in range(len(intervaller)-1):
            xs = np.linspace(intervaller[i]+0.05, intervaller[i+1]-0.05, 300)
            ys = f_num(xs)
            ys[np.abs(ys) > 100] = np.nan
            ax_g.plot(xs, ys, 'k', lw=2)
        
        for b in b_vals:
            ax_g.axvline(b, color='red', linestyle='--', alpha=0.6)

        ax_g.set_ylim(y_mid - y_r*v_zoom, y_mid + y_r*v_zoom)
        ax_g.axhline(0, color='gray', lw=1, alpha=0.3)
        ax_g.axvline(0, color='gray', lw=1, alpha=0.3)
        ax_g.grid(True, alpha=0.2)
        st.pyplot(fig_g)

    # Nedlasting
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    st.sidebar.download_button("📥 Last ned PNG", buf.getvalue(), "analyse.png")

except Exception as e:
    st.error(f"Feil: {e}")
