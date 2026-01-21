import streamlit as st
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import io

# Sideoppsett
st.set_page_config(page_title="Matte-Analyse: Drøfting", layout="wide")
st.title("📈 Funksjonsdrøfting med Automatisk Tolkning")

# --- SIDEBAR: KONTROLLPANEL ---
with st.sidebar:
    st.header("1. Definer funksjon")
    input_f = st.text_input("Skriv inn f(x):", "x**3 - 3*x**2 + 2")
    
    st.header("2. Velg Analyse-nivå")
    analyse_nivå = st.radio(
        "Hva skal analyseres?",
        ["Originalfunksjon f(x)", "Førstederivert f'(x)", "Andrederivert f''(x)"]
    )
    
    st.header("3. Lærer-modus")
    skjul_info = st.checkbox("Skjul matematisk uttrykk og fasit", value=False)
    skjul_faktorer = st.checkbox("Skjul faktornavn", value=False)
    skjul_x = st.checkbox("Skjul x-verdier", value=False)
    
    st.header("4. Design")
    farge_tema = st.selectbox("Farger", ["Svart", "Blå/Rød"])

# --- MATEMATISK LOGIKK ---
try:
    x = sp.symbols('x')
    f_orig = sp.sympify(input_f)
    f1 = sp.diff(f_orig, x)
    f2 = sp.diff(f1, x)
    
    # Velg mål-funksjon
    if analyse_nivå == "Originalfunksjon f(x)":
        target_f, label_prefix, f_grad = f_orig, "f(x)", 0
    elif analyse_nivå == "Førstederivert f'(x)":
        target_f, label_prefix, f_grad = f1, "f'(x)", 1
    else:
        target_f, label_prefix, f_grad = f2, "f''(x)", 2

    # Faktorisering
    target_f_faktorisert = sp.factor(target_f)
    teller, nevner = sp.fraction(target_f_faktorisert)
    
    # Finn alle kritiske x-verdier (nullpunkter og bruddpunkter)
    nullpunkter = sp.solve(teller, x)
    bruddpunkter = sp.solve(nevner, x)
    alle_kritiske = sorted(list(set([sp.re(p) for p in (nullpunkter + bruddpunkter) if p.is_real])), key=float)

    # --- TEGNEFUNKSJON ---
    def tegn_skjema():
        margin = 2.0
        x_min = float(alle_kritiske[0]) - margin if alle_kritiske else -5
        x_max = float(alle_kritiske[-1]) + margin if alle_kritiske else 5
        plot_pts = sorted(list(set([x_min, x_max] + [float(val) for val in alle_kritiske])))
        
        # Finn faktorer for visning
        t_fakt, n_fakt = sp.factor_list(teller)[1], sp.factor_list(nevner)[1]
        konst = sp.factor_list(teller)[0] / sp.factor_list(nevner)[0]
        faktorer_linjer = []
        if abs(konst - 1) > 1e-9: faktorer_linjer.append(konst)
        for fakt, eksp in t_fakt: faktorer_linjer.append(fakt**eksp)
        for fakt, eksp in n_fakt: faktorer_linjer.append(fakt**eksp)
        
        rader = faktorer_linjer + [target_f]
        fig, ax = plt.subplots(figsize=(12, len(rader) * 0.9))
        ax.xaxis.set_ticks_position('top')
        
        for idx, uttrykk in enumerate(reversed(rader)):
            y = idx
            label = label_prefix if idx == 0 else (f"Faktor {len(rader)-idx}" if skjul_faktorer else f"${sp.latex(uttrykk)}$")
            ax.text(x_min - 0.2, y, label, va='center', ha='right', fontsize=12)
            
            for i in range(len(plot_pts)-1):
                mid = (plot_pts[i] + plot_pts[i+1]) / 2
                verdi = uttrykk.subs(x, mid)
                pos = verdi > 0
                ls, c = ('-', 'black') if pos else ('--', 'black')
                if farge_tema == "Blå/Rød": c = 'blue' if pos else 'red'
                ax.plot([plot_pts[i], plot_pts[i+1]], [y, y], linestyle=ls, color=c, lw=2.5)

            for p in alle_kritiske:
                ax.axvline(float(p), color='gray', lw=0.6, linestyle=':', alpha=0.5)
                if idx == 0:
                    sym = 'X' if any(sp.simplify(p-b)==0 for b in bruddpunkter) else '0'
                    ax.text(float(p), y, sym, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none'))
                else:
                    try: 
                        if abs(uttrykk.subs(x, p)) < 1e-9: ax.text(float(p), y, '0', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none'))
                    except: pass

        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        if not skjul_x:
            ax.set_xticks([float(v) for v in alle_kritiske])
            ax.set_xticklabels([f"${sp.latex(v)}$" for v in alle_kritiske])
        else: ax.set_xticks([])
        ax.spines['top'].set_visible(True)
        ax.spines[['bottom', 'left', 'right']].set_visible(False)
        ax.get_yaxis().set_visible(False)
        return fig

    # --- VISNING ---
    if not skjul_info:
        st.latex(f"{label_prefix} = {sp.latex(target_f_faktorisert)}")
    
    st.pyplot(tegn_skjema())

    # --- DYNAMISK ANALYSE-MODUL ---
    if not skjul_info:
        st.divider()
        st.subheader("📝 Automatisk Analyse & Tolkning")
        
        # 1. Bruddpunkter (Gjelder alle)
        if bruddpunkter:
            st.warning(f"**Bruddpunkter:** Funksjonen er ikke definert for $x = {sp.latex(bruddpunkter)}$. Her har grafen vertikale asymptoter.")

        # 2. Spesifikk analyse basert på nivå
        if f_grad == 0: # f(x)
            st.write(f"**Nullpunkter:** Grafen skjærer x-aksen i $x = {sp.latex(nullpunkter)}$.")
            st.write(f"**Positiv/Negativ:** Se hvor linjen for $f(x)$ er hel (over x-aksen) eller stiplet (under x-aksen).")

        elif f_grad == 1: # f'(x)
            st.write("**Monotoniegenskaper (Stigning):**")
            for p in nullpunkter:
                if p.is_real:
                    # Sjekk fortegn før og etter for å finne type punkt
                    v_for = f1.subs(x, p - 0.01)
                    v_etter = f1.subs(x, p + 0.01)
                    y_val = f_orig.subs(x, p)
                    
                    if v_for > 0 and v_etter < 0:
                        st.success(f"I $x = {sp.latex(p)}$ har vi et **toppunkt**: $({sp.latex(p)}, {sp.latex(y_val)})$")
                    elif v_for < 0 and v_etter > 0:
                        st.success(f"I $x = {sp.latex(p)}$ har vi et **bunnpunkt**: $({sp.latex(p)}, {sp.latex(y_val)})$")
                    elif (v_for > 0 and v_etter > 0) or (v_for < 0 and v_etter < 0):
                        st.info(f"I $x = {sp.latex(p)}$ har vi et **terassepunkt**: $({sp.latex(p)}, {sp.latex(y_val)})$")

        elif f_grad == 2: # f''(x)
            st.write("**Krumning og Vendepunkter:**")
            for p in nullpunkter:
                if p.is_real:
                    v_for = f2.subs(x, p - 0.01)
                    v_etter = f2.subs(x, p + 0.01)
                    if (v_for * v_etter) < 0: # Fortegnsskifte
                        st.success(f"I $x = {sp.latex(p)}$ har vi et **vendepunkt**. Grafen skifter krumning.")
            st.write("- **Hel linje:** Hulsiden vender opp (konveks / 'smiler').")
            st.write("- **Stiplet linje:** Hulsiden vender ned (konkav / 'sur').")

    # Nedlasting
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    st.sidebar.download_button("📥 Last ned skjema (PNG)", buf.getvalue(), "analyse.png")

except Exception as e:
    st.error(f"Feil: {e}")
