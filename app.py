import streamlit as st
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import io

# Sideoppsett
st.set_page_config(page_title="Matte-Analyse: Proff-graf", layout="wide")
st.title("📈 Funksjonsanalyse med asymptoter")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Definer funksjon")
    input_f = st.text_input("Skriv inn f(x):", "(x**2 - 4) / (x - 1)")
    
    st.header("2. Analyse-nivå")
    nivå = st.radio("Nivå:", ["f(x)", "f'(x)", "f''(x)"])
    
    st.header("3. Visning")
    vis_konstant = st.checkbox("Vis konstante tall", value=True)
    skjul_var = st.checkbox("Skjul faktorer med x", value=False)
    v_info = not st.checkbox("Skjul fasit", value=False)
    v_x_akse = not st.checkbox("Skjul x-verdier", value=False)
    
    st.header("4. Graf-innstillinger")
    v_graf = st.checkbox("Vis grafen", value=True)
    zoom_faktor = st.slider("Vertikal zoom", 1.1, 8.0, 2.0)

# --- MATEMATIKK ---
try:
    x = sp.symbols('x')
    f_orig = sp.sympify(input_f)
    f1 = sp.diff(f_orig, x)
    f2 = sp.diff(f1, x)
    
    if nivå == "f(x)": target, label, grad = f_orig, "f(x)", 0
    elif nivå == "f'(x)": target, label, grad = f1, "f'(x)", 1
    else: target, label, grad = f2, "f''(x)", 2

    target_fakt = sp.factor(target)
    t_del, n_del = sp.fraction(target_fakt)
    nullpunkter = sp.solve(t_del, x)
    bruddpunkter = sp.solve(n_del, x)
    stasjonære = sp.solve(f1, x)
    
    kritiske = sorted(list(set([sp.re(p) for p in (nullpunkter + bruddpunkter) if p.is_real])), key=float)

    # --- TEGNEFUNKSJON SKJEMA ---
    def tegn_skjema():
        margin = 1.5
        x_min = float(kritiske[0]) - margin if kritiske else -5
        x_max = float(kritiske[-1]) + margin if kritiske else 5
        pts = sorted(list(set([x_min, x_max] + [float(v) for v in kritiske])))
        
        t_list, n_list = sp.factor_list(t_del)[1], sp.factor_list(n_del)[1]
        konst = sp.factor_list(t_del)[0] / sp.factor_list(n_del)[0]
        
        rader = []
        if abs(konst - 1) > 1e-9: rader.append(('k', konst))
        for fkt, ek in t_list: rader.append(('v', fkt**ek))
        for fkt, ek in n_list: rader.append(('v', fkt**ek))
        rader.append(('t', target))
        
        fig, ax = plt.subplots(figsize=(12, len(rader) * 1.1))
        ax.xaxis.set_ticks_position('top')

        for idx, (type, uttr) in enumerate(reversed(rader)):
            y = idx
            if type == 't': n = label
            elif type == 'k': n = f"${sp.latex(uttr)}$" if vis_konstant else "Konstant"
            else: n = f"Faktor {len(rader)-idx}" if skjul_var else f"${sp.latex(uttr)}$"
            
            ax.text(x_min - 0.2, y, n, va='center', ha='right', fontsize=13)
            for i in range(len(pts)-1):
                m = (pts[i] + pts[i+1]) / 2
                res = uttr.subs(x, m) if hasattr(uttr, 'subs') else uttr
                pos = res > 0
                ax.plot([pts[i], pts[i+1]], [y, y], '-' if pos else '--', color='black', lw=2.5)
                if type == 't' and v_info:
                    if grad == 1: ax.text(m, y-0.35, r"$\nearrow$" if pos else r"$\searrow$", ha='center', fontsize=20, color='gray')
                    elif grad == 2: ax.text(m, y-0.35, r"$\cup$" if pos else r"$\cap$", ha='center', fontsize=20, color='gray')

            for p in kritiske:
                p_v = float(p)
                ax.axvline(p_v, color='gray', lw=0.6, linestyle=':', alpha=0.5)
                if type == 't':
                    ax.text(p_v, y, 'X' if any(sp.simplify(p-b)==0 for b in bruddpunkter) else '0', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none'))
                elif type == 'v' and abs(float(uttr.subs(x, p))) < 1e-9:
                    ax.text(p_v, y, '0', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none'))

        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(-0.8, len(rader) + 0.4)
        if v_x_akse:
            ax.set_xticks([float(v) for v in kritiske])
            ax.set_xticklabels([f"${sp.latex(v)}$" for v in kritiske])
        else: ax.set_xticks([])
        ax.spines['top'].set_visible(True)
        ax.spines[['bottom', 'left', 'right']].set_visible(False)
        ax.get_yaxis().set_visible(False)
        return fig

    st.pyplot(tegn_skjema())

    # --- GRAF MED ASYMPTOTER ---
    if v_graf:
        st.divider()
        st.subheader("Grafisk fremstilling med asymptoter")
        
        # 1. Bestem x-området
        x_min_val = float(min(kritiske)) - 4 if kritiske else -7
        x_max_val = float(max(kritiske)) + 4 if kritiske else 7
        
        # 2. Finn y-skala (ekstremalpunkter)
        y_vals_for_scale = []
        for p in stasjonære:
            if p.is_real:
                val = float(f_orig.subs(x, p))
                if np.isfinite(val): y_vals_for_scale.append(val)
        
        if not y_vals_for_scale: y_vals_for_scale = [-2, 2]
        y_mid = (max(y_vals_for_scale) + min(y_vals_for_scale)) / 2
        y_range = max(y_vals_for_scale) - min(y_vals_for_scale)
        if y_range < 0.1: y_range = 4.0
        
        fig_g, ax_g = plt.subplots(figsize=(10, 5))
        
        # 3. SEGMENTERT PLOTTING: Splitt x-aksen ved bruddpunkter
        brudd_vals = sorted([float(p) for p in bruddpunkter if p.is_real])
        grenser = [x_min_val] + brudd_vals + [x_max_val]
        
        f_numpy = sp.lambdify(x, f_orig, "numpy")
        
        for i in range(len(grenser)-1):
            x_start_seg = grenser[i] + 0.05 # Liten margin fra asymptote
            x_slutt_seg = grenser[i+1] - 0.05
            if x_start_seg < x_slutt_seg:
                x_seg = np.linspace(x_start_seg, x_slutt_seg, 200)
                y_seg = f_numpy(x_seg)
                ax_g.plot(x_seg, y_seg, 'k', lw=2)

        # 4. Tegn vertikale asymptoter
        for b in brudd_vals:
            ax_g.axvline(b, color='red', linestyle='--', lw=1.5, label='Asymptote' if b == brudd_vals[0] else "")

        # 5. Styling
        ax_g.set_ylim(y_mid - (y_range * zoom_faktor), y_mid + (y_range * zoom_faktor))
        ax_g.set_xlim(x_min_val, x_max_val)
        ax_g.axhline(0, color='black', lw=0.8, alpha=0.3)
        ax_g.axvline(0, color='black', lw=0.8, alpha=0.3)
        ax_g.grid(True, alpha=0.2)
        if brudd_vals: ax_g.legend()
        
        st.pyplot(fig_g)

except Exception as e:
    st.error(f"Feil: {e}")
