import streamlit as st
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import io

# Sideoppsett
st.set_page_config(page_title="Oppgavegenerator: Fortegnsskjema", layout="wide")
st.title("📝 Generator for Fortegnsskjema-oppgaver")

# --- SIDEBAR: KONTROLLPANEL ---
with st.sidebar:
    st.header("1. Definer funksjon")
    input_f = st.text_input("Funksjon f(x):", "(2*x - 4)*(x + 3) / (x - 1)")
    
    st.header("2. Lærer-modus (Skjul elementer)")
    skjul_funksjon = st.checkbox("Skjul funksjonsuttrykk (overskrift)", value=False)
    skjul_faktorer = st.checkbox("Skjul faktornavn (venstre side)", value=False)
    skjul_x_verdier = st.checkbox("Skjul verdier på x-aksen", value=False)
    skjul_derivert = st.checkbox("Skjul derivasjonsanalyse (bunnen)", value=True)
    
    st.header("3. Design")
    farge_tema = st.selectbox("Linjefarge", ["Svart", "Blå/Rød"], index=0)

# --- HOVEDLOGIKK ---
try:
    x = sp.symbols('x')
    f = sp.sympify(input_f)
    
    # Faktorisering og klargjøring
    f_faktorisert = sp.factor(f)
    teller, nevner = sp.fraction(f_faktorisert)
    
    t_faktorer_list = sp.factor_list(teller)[1]
    n_faktorer_list = sp.factor_list(nevner)[1]
    konstant = sp.factor_list(teller)[0] / sp.factor_list(nevner)[0]

    alle_faktorer = []
    if abs(konstant - 1) > 1e-9:
        alle_faktorer.append(konstant)
    for fakt, eksp in t_faktorer_list: alle_faktorer.append(fakt**eksp)
    for fakt, eksp in n_faktorer_list: alle_faktorer.append(fakt**eksp)

    nullpunkter = sp.solve(teller, x)
    bruddpunkter = sp.solve(nevner, x)
    kritiske_x_sym = sorted(list(set([p for p in (nullpunkter + bruddpunkter) if p.is_real])), key=lambda v: float(v))

    # --- FUNKSJON FOR Å TEGNE SKJEMA ---
    def tegn_skjema():
        margin = 2.0
        x_min = float(kritiske_x_sym[0]) - margin if kritiske_x_sym else -5
        x_max = float(kritiske_x_sym[-1]) + margin if kritiske_x_sym else 5
        plot_pts = sorted(list(set([x_min, x_max] + [float(val) for val in kritiske_x_sym])))
        
        rader = alle_faktorer + [f]
        fig, ax = plt.subplots(figsize=(12, len(rader) * 1.0))
        
        # Plasser x-aksen på toppen
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

        for idx, uttrykk in enumerate(reversed(rader)):
            y = idx
            if skjul_faktorer:
                label = "f(x)" if idx == 0 else f"Faktor {len(rader)-idx}"
            else:
                label = "f(x)" if idx == 0 else f"${sp.latex(uttrykk)}$"
            
            ax.text(x_min - 0.2, y, label, va='center', ha='right', fontsize=14)
            
            for i in range(len(plot_pts)-1):
                x1, x2 = plot_pts[i], plot_pts[i+1]
                mid = (x1 + x2) / 2
                try:
                    verdi = uttrykk.subs(x, mid)
                    pos = verdi > 0
                except: pos = float(uttrykk) > 0

                ls, farge = ('-', 'black') if pos else ('--', 'black')
                if farge_tema == "Blå/Rød":
                    farge = 'blue' if pos else 'red'
                
                ax.plot([x1, x2], [y, y], linestyle=ls, color=farge, lw=2.5)

            for p_sym in kritiske_x_sym:
                p_val = float(p_sym)
                ax.axvline(p_val, color='gray', lw=0.8, linestyle=':', alpha=0.7)
                
                if idx == 0:
                    is_brudd = any([sp.simplify(p_sym - b) == 0 for b in bruddpunkter])
                    ax.text(p_val, y, 'X' if is_brudd else '0', ha='center', va='center', 
                            fontsize=14, bbox=dict(facecolor='white', edgecolor='none', pad=3))
                else:
                    try:
                        if abs(float(uttrykk.subs(x, p_sym))) < 1e-9:
                             ax.text(p_val, y, '0', ha='center', va='center', fontsize=14, 
                                     bbox=dict(facecolor='white', edgecolor='none', pad=3))
                    except: pass

        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(-0.5, len(rader) - 0.2)
        
        if not skjul_x_verdier:
            ax.set_xticks([float(v) for v in kritiske_x_sym])
            ax.set_xticklabels([f"${sp.latex(v)}$" for v in kritiske_x_sym], fontsize=13)
        else:
            ax.set_xticks([])

        ax.spines['top'].set_visible(True)
        ax.spines[['bottom', 'right', 'left']].set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.tight_layout()
        return fig

    # Vis analyse og plott
    if not skjul_funksjon:
        st.latex(f"f(x) = {sp.latex(f_faktorisert)}")
    
    fig = tegn_skjema()
    st.pyplot(fig)

    # --- NEDLASTING ---
    png_buffer = io.BytesIO()
    fig.savefig(png_buffer, format='png', dpi=300, bbox_inches='tight')
    
    with st.sidebar:
        st.header("4. Last ned")
        st.download_button("📥 Last ned PNG", data=png_buffer.getvalue(), file_name="fortegn.png", mime="image/png")

    if not skjul_derivert:
        st.divider()
        f1 = sp.diff(f, x)
        st.write("**Derivert:**")
        st.latex(f"f'(x) = {sp.latex(sp.simplify(f1))}")

except Exception as e:
    st.error(f"Feil i uttrykket: {e}")
