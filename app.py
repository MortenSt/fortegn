import streamlit as st
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import io

# Sideoppsett
st.set_page_config(page_title="Matte-Analyse", layout="wide")
st.title("📈 Avansert Funksjonsanalyse")

# --- SIDEBAR: KONTROLLPANEL ---
with st.sidebar:
    st.header("1. Definer funksjon")
    input_f = st.text_input("Skriv inn f(x):", "x**3 - 3*x**2 + 2")
    
    with st.expander("ℹ️ Inntastingsguide"):
        st.markdown("""
        * **Potens:** `x**2`
        * **Gange:** `2*x`
        * **Brøk:** `(x+1)/(x-1)`
        * **Kvadratrot:** `sqrt(x)`
        * **Eulers tall:** `exp(x)`
        """)

    st.header("2. Analyse-nivå")
    nivå = st.radio("Hva skal analyseres?", ["f(x)", "f'(x)", "f''(x)"])
    
    st.header("3. Visning (Lærer-modus)")
    v_info = not st.checkbox("Skjul fasit og uttrykk", value=False)
    v_faktorer = not st.checkbox("Skjul faktornavn", value=False)
    v_x_akse = not st.checkbox("Skjul x-verdier", value=False)
    v_graf = st.checkbox("Vis grafen til f(x)", value=True)
    
    st.header("4. Design")
    farge_valg = st.selectbox("Fargetema", ["Svart", "Blå/Rød"])

# --- MATEMATISK LOGIKK ---
try:
    x = sp.symbols('x')
    f_orig = sp.sympify(input_f)
    f1 = sp.diff(f_orig, x)
    f2 = sp.diff(f1, x)
    
    # Velg målfunksjon basert på nivå
    if nivå == "f(x)":
        target, label, grad = f_orig, "f(x)", 0
    elif nivå == "f'(x)":
        target, label, grad = f1, "f'(x)", 1
    else:
        target, label, grad = f2, "f''(x)", 2

    # Faktorisering
    target_fakt = sp.factor(target)
    t, n = sp.fraction(target_fakt)
    
    # Nullpunkter og bruddpunkter
    nullpunkter = sp.solve(t, x)
    bruddpunkter = sp.solve(n, x)
    kritiske = sorted(list(set([sp.re(p) for p in (nullpunkter + bruddpunkter) if p.is_real])), key=float)

    # --- TEGNEFUNKSJON ---
    def tegn_skjema(target_f, kritiske_pkt, label_text, vis_fakt, vis_x, farge_tema, analyse_grad):
        margin = 2.0
        x_min = float(kritiske_pkt[0]) - margin if kritiske_pkt else -5
        x_max = float(kritiske_pkt[-1]) + margin if kritiske_pkt else 5
        pts = sorted(list(set([x_min, x_max] + [float(v) for v in kritiske_pkt])))
        
        # Hent faktorer
        t_del, n_del = sp.fraction(sp.factor(target_f))
        t_list, n_list = sp.factor_list(t_del)[1], sp.factor_list(n_del)[1]
        konst = sp.factor_list(t_del)[0] / sp.factor_list(n_del)[0]
        
        linjer = []
        if abs(konst - 1) > 1e-9: linjer.append(konst)
        for fkt, ek in t_list: linjer.append(fkt**ek)
        for fkt, ek in n_list: linjer.append(fkt**ek)
        
        alle_rader = linjer + [target_f]
        fig, ax = plt.subplots(figsize=(12, len(alle_rader) * 1.2))
        
        # X-akse på topp
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

        for idx, uttr in enumerate(reversed(alle_rader)):
            y_pos = idx
            # Etikett til venstre
            if idx == 0:
                rad_navn = label_text
            else:
                rad_navn = f"Faktor {len(alle_rader)-idx}" if not vis_fakt else f"${sp.latex(uttr)}$"
            
            ax.text(x_min - 0.2, y_pos, rad_navn, va='center', ha='right', fontsize=13)
            
            # Tegn intervaller
            for i in range(len(pts)-1):
                m = (pts[i] + pts[i+1]) / 2
                res = uttr.subs(x, m)
                is_pos = res > 0
                
                ls = '-' if is_pos else '--'
                c = 'black'
                if farge_tema == "Blå/Rød": c = 'blue' if is_pos else 'red'
                
                ax.plot([pts[i], pts[i+1]], [y_pos, y_pos], linestyle=ls, color=c, lw=2.5)

                # Illustrasjoner (Piler og krumning) kun på hovedlinjen
                if idx == 0 and v_info:
                    if analyse_grad == 1: # f'
                        sym = r"$\nearrow$" if is_pos else r"$\searrow$"
                        ax.text(m, y_pos - 0.35, sym, ha='center', fontsize=20, color='gray')
                    elif analyse_grad == 2: # f''
                        sym = r"$\cup$" if is_pos else r"$\cap$"
                        ax.text(m, y_pos - 0.35, sym, ha='center', fontsize=20, color='gray')

            # Markører (0 og X)
            for p in kritiske_pkt:
                p_v = float(p)
                ax.axvline(p_v, color='gray', lw=0.6, linestyle=':', alpha=0.5)
                if idx == 0:
                    er_b = any(sp.simplify(p-b)==0 for b in bruddpunkter)
                    ax.text(p_v, y_pos, 'X' if er_b else '0', ha='center', va='center', 
                            bbox=dict(facecolor='white', edgecolor='none', pad=2))

        # --- MELLOMROM: Justerer ylim for å gi plass mellom x-akse og øverste faktor ---
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(-0.8, len(alle_rader) + 0.4) 

        if vis_x:
            ax.set_xticks([float(v) for v in kritiske_pkt])
            ax.set_xticklabels([f"${sp.latex(v)}$" for v in kritiske_pkt], fontsize=12)
        else:
            ax.set_xticks([])

        ax.spines['top'].set_visible(True)
        ax.spines[['bottom', 'left', 'right']].set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.tight_layout()
        return fig

    # --- VISNING I APP ---
    if v_info:
        st.latex(f"{label} = {sp.latex(target_fakt)}")
    
    skjema_fig = tegn_skjema(target, kritiske, label, v_faktorer, v_x_akse, farge_valg, grad)
    st.pyplot(skjema_fig)

    # --- GRAF ---
    if v_graf:
        st.divider()
        st.subheader("Grafen til f(x)")
        m_g = 4.0
        x_g = np.linspace(float(min(kritiske))-m_g if kritiske else -7, float(max(kritiske))+m_g if kritiske else 7, 500)
        f_n = sp.lambdify(x, f_orig, "numpy")
        try:
            y_g = f_n(x_g)
            fig_g, ax_g = plt.subplots(figsize=(10, 4))
            ax_g.plot(x_g, y_g, 'k', lw=2)
            ax_g.axhline(0, color='gray', lw=1)
            ax_g.axvline(0, color='gray', lw=1)
            ax_g.grid(True, alpha=0.2)
            st.pyplot(fig_g)
        except: st.warning("Kunne ikke tegne grafen.")

    # --- TOLKNING ---
    if v_info:
        st.divider()
        st.subheader("📝 Analyse")
        if grad == 1:
            st.write("**Stigningsegenskaper:**")
            for p in [pt for pt in nullpunkter if pt.is_real]:
                v1, v2 = f1.subs(x, p-0.01), f1.subs(x, p+0.01)
                y_v = f_orig.subs(x, p)
                if v1 > 0 and v2 < 0: st.success(f"Toppunkt i $({sp.latex(p)}, {sp.latex(y_v)})$")
                elif v1 < 0 and v2 > 0: st.success(f"Bunnpunkt i $({sp.latex(p)}, {sp.latex(y_v)})$")
        elif grad == 2:
            st.write("**Krumning:**")
            st.write("- $\cup$ betyr hulsiden opp (konveks)")
            st.write("- $\cap$ betyr hulsiden ned (konkav)")

    # Nedlasting
    buf = io.BytesIO()
    skjema_fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    st.sidebar.download_button("📥 Last ned PNG", buf.getvalue(), "fortegnsskjema.png", "image/png")

except Exception as e:
    st.error(f"Feil i inntasting: {e}. Husk å bruke `**` for potens og `*` for gange.")
