import streamlit as st
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import io

# Sideoppsett
st.set_page_config(page_title="Matte-Analyse: Drøfting", layout="wide")
st.title("📈 Funksjonsdrøfting med krumning og stigning")

# --- SIDEBAR: KONTROLLPANEL ---
with st.sidebar:
    st.header("1. Definer funksjon")
    input_f = st.text_input("Skriv inn f(x):", "x**3 - 3*x**2 + 2")
    
    with st.expander("ℹ️ Hvordan skrive inn matte?"):
        st.markdown("- **Potens:** `x**2`\n- **Gange:** `2*x`\n- **Brøk:** `(x+1)/(x-1)`\n- **Kvadratrot:** `sqrt(x)`")

    st.header("2. Velg Analyse-nivå")
    analyse_nivå = st.radio("Hva skal analyseres?", ["Originalfunksjon f(x)", "Førstederivert f'(x)", "Andrederivert f''(x)"])
    
    st.header("3. Lærer-modus")
    skjul_info = st.checkbox("Skjul matematisk uttrykk og fasit", value=False)
    vis_graf = st.checkbox("Vis selve grafen til f(x)", value=True)
    
    st.header("4. Design")
    farge_tema = st.selectbox("Farger", ["Svart", "Blå/Rød"])

# --- MATEMATISK LOGIKK ---
try:
    x = sp.symbols('x')
    f_orig = sp.sympify(input_f)
    f1 = sp.diff(f_orig, x)
    f2 = sp.diff(f1, x)
    
    if analyse_nivå == "Originalfunksjon f(x)":
        target_f, label_prefix, f_grad = f_orig, "f(x)", 0
    elif analyse_nivå == "Førstederivert f'(x)":
        target_f, label_prefix, f_grad = f1, "f'(x)", 1
    else:
        target_f, label_prefix, f_grad = f2, "f''(x)", 2

    target_f_faktorisert = sp.factor(target_f)
    teller, nevner = sp.fraction(target_f_faktorisert)
    
    nullpunkter = sp.solve(teller, x)
    bruddpunkter = sp.solve(nevner, x)
    alle_kritiske = sorted(list(set([sp.re(p) for p in (nullpunkter + bruddpunkter) if p.is_real])), key=float)

    # --- TEGNEFUNKSJON FOR SKJEMA ---
    def tegn_skjema():
        margin = 2.0
        x_min = float(alle_kritiske[0]) - margin if alle_kritiske else -5
        x_max = float(alle_kritiske[-1]) + margin if alle_kritiske else 5
        plot_pts = sorted(list(set([x_min, x_max] + [float(val) for val in alle_kritiske])))
        
        t_fakt, n_fakt = sp.factor_list(teller)[1], sp.factor_list(nevner)[1]
        konst = sp.factor_list(teller)[0] / sp.factor_list(nevner)[0]
        faktorer_linjer = []
        if abs(konst - 1) > 1e-9: faktorer_linjer.append(konst)
        for fakt, eksp in t_fakt: faktorer_linjer.append(fakt**eksp)
        for fakt, eksp in n_fakt: faktorer_linjer.append(fakt**eksp)
        
        rader = faktorer_linjer + [target_f]
        fig, ax = plt.subplots(figsize=(12, len(rader) * 1.2))
        ax.xaxis.set_ticks_position('top')
        
        for idx, uttrykk in enumerate(reversed(rader)):
            y = idx
            label = label_prefix if idx == 0 else f"${sp.latex(uttrykk)}$"
            ax.text(x_min - 0.2, y, label, va='center', ha='right', fontsize=13)
            
            for i in range(len(plot_pts)-1):
                mid = (plot_pts[i] + plot_pts[i+1]) / 2
                try:
                    verdi = uttrykk.subs(x, mid)
                    pos = verdi > 0
                except: pos = float(uttrykk) > 0

                ls, c = ('-', 'black') if pos else ('--', 'black')
                if farge_tema == "Blå/Rød": c = 'blue' if pos else 'red'
                ax.plot([plot_pts[i], plot_pts[i+1]], [y, y], linestyle=ls, color=c, lw=2.5)

                # --- NYTT: Illustrasjon av stigning og krumning ---
                if idx == 0 and not skjul_info:
                    symbol = ""
                    if f_grad == 1: # f'(x) -> Stigning
                        symbol = r"$\nearrow$" if pos else r"$\searrow$"
                    elif f_grad == 2: # f''(x) -> Krumning
                        symbol = r"$\cup$" if pos else r"$\cap$"
                    
                    if symbol:
                        ax.text(mid, y - 0.3, symbol, ha='center', fontsize=18, color='gray')

            for p in alle_kritiske:
                ax.axvline(float(p), color='gray', lw=0.6, linestyle=':', alpha=0.5)
                if idx == 0:
                    sym = 'X' if any(sp.simplify(p-b)==0 for b in bruddpunkter) else '0'
                    ax.text(float(p), y, sym, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none'))

        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(-0.7, len(rader) + 0.3) # Mer plass til x-akse og symboler
        if not skjul_x:
            ax.set_xticks([float(v) for v in alle_kritiske])
            ax.set_xticklabels([f"${sp.latex(v)}$" for v in alle_kritiske], fontsize=12)
        else: ax.set_xticks([])
        
        ax.spines['top'].set_visible(True)
        ax.spines[['bottom', 'left', 'right']].set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.tight_layout()
        return fig

    # --- VISNING ---
    if not skjul_info:
        st.latex(f"{label_prefix} = {sp.latex(target_f_faktorisert)}")
    
    st.pyplot(tegn_skjema())

    # --- NYTT: GRAFTEGNING ---
    if vis_graf:
        st.subheader("Grafisk fremstilling av f(x)")
        margin = 3.0
        x_min_f = float(alle_kritiske[0]) - margin if alle_kritiske else -5
        x_max_f = float(alle_kritiske[-1]) + margin if alle_kritiske else 5
        
        # Lag numpy-funksjon for rask plotting
        f_func = sp.lambdify(x, f_orig, "numpy")
        x_vals = np.linspace(x_min_f, x_max_f, 400)
        try:
            y_vals = f_func(x_vals)
            
            fig_graf, ax_graf = plt.subplots(figsize=(10, 4))
            ax_graf.plot(x_vals, y_vals, color='black', lw=2, label="f(x)")
            ax_graf.axhline(0, color='gray', lw=1)
            ax_graf.axvline(0, color='gray', lw=1)
            ax_graf.grid(True, alpha=0.3)
            
            # Marker nullpunkter og ekstremalpunkter på grafen
            for p in alle_kritiske:
                p_val = float(p)
                try:
                    y_p = float(f_orig.subs(x, p_val))
                    ax_graf.plot(p_val, y_p, 'ro') # Rød prikk
                except: pass
            
            st.pyplot(fig_graf)
        except:
            st.info("Grafen kunne ikke tegnes (kanskje pga. bruddpunkter eller komplekse tall).")

    # --- ANALYSE-MODUL ---
    if not skjul_info:
        st.divider()
        st.subheader("📝 Tolkning")
        if f_grad == 1:
            st.write("**Stigning ($\nearrow$):** Grafen går oppover når $f'(x)$ er positiv (hel linje).")
            st.write("**Synking ($\searrow$):** Grafen går nedover når $f'(x)$ er negativ (stiplet linje).")
        elif f_grad == 2:
            st.write("**Krumning opp ($\cup$):** Grafen er 'smilende' når $f''(x)$ er positiv.")
            st.write("**Krumning ned ($\cap$):** Grafen er 'sur' når $f''(x)$ er negativ.")

    # Nedlasting
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    st.sidebar.download_button("📥 Last ned bilde", buf.getvalue(), "fortegn.png")

except Exception as e:
    st.error(f"Feil: {e}")
