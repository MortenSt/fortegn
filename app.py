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
    # Setter en litt mer avansert standardfunksjon som demonstrerer brudd og nullpunkter godt
    input_f = st.text_input("Funksjon f(x):", "(2*x - 4)*(x + 3) / (x - 1)")
    
    st.header("2. Lærer-modus (Skjul elementer)")
    skjul_funksjon = st.checkbox("Skjul funksjonsuttrykk (overskrift)", value=False)
    skjul_faktorer = st.checkbox("Skjul faktornavn (venstre side)", value=False)
    skjul_x_verdier = st.checkbox("Skjul verdier på x-aksen", value=False)
    skjul_derivert = st.checkbox("Skjul derivasjonsanalyse (bunnen)", value=True)
    
    st.header("3. Design")
    farge_tema = st.selectbox("Linjefarge", ["Svart", "Blå/Rød"], index=0)
    st.info("Tips: 'Svart' er best for utskrift. 'Blå/Rød' er best for skjermvisning.")

# --- MATEMATISK LOGIKK ---
try:
    x = sp.symbols('x')
    f = sp.sympify(input_f)
    
    # Faktorisering
    f_faktorisert = sp.factor(f)
    teller, nevner = sp.fraction(f_faktorisert)
    
    # Hent ut individuelle faktorer for tegning
    t_faktorer_list = sp.factor_list(teller)[1]
    n_faktorer_list = sp.factor_list(nevner)[1]
    konstant = sp.factor_list(teller)[0] / sp.factor_list(nevner)[0]

    alle_faktorer = []
    if abs(konstant - 1) > 1e-9: # Legg til konstant hvis den ikke er 1
        alle_faktorer.append(konstant)
    for fakt, eksp in t_faktorer_list: alle_faktorer.append(fakt**eksp)
    for fakt, eksp in n_faktorer_list: alle_faktorer.append(fakt**eksp)

    # Kritiske punkter
    nullpunkter = sp.solve(teller, x)
    bruddpunkter = sp.solve(nevner, x)
    # Filtrer kun reelle løsninger og konverter til sortert liste av unike tall
    kritiske_x_sym = sorted(list(set([p for p in (nullpunkter + bruddpunkter) if p.is_real])), key=lambda v: float(v))

    # --- VISNING ---
    if not skjul_funksjon:
        st.subheader("Analyse av funksjon")
        st.latex(f"f(x) = {sp.latex(f_faktorisert)}")
    else:
        st.subheader("Oppgave: Analyser funksjonen basert på skjemaet")
        st.write("*(Funksjonsuttrykket er skjult)*")

    # --- PLOTTING AV FORTEGNSSKJEMA ---
    def tegn_skjema():
        margin = 2.0
        # Håndter tilfeller uten kritiske punkter
        x_min = float(kritiske_x_sym[0]) - margin if kritiske_x_sym else -5
        x_max = float(kritiske_x_sym[-1]) + margin if kritiske_x_sym else 5
        
        # Lag plottepunkter: start, slutt, og alle kritiske punkter
        plot_pts = sorted(list(set([x_min, x_max] + [float(val) for val in kritiske_x_sym])))
        
        rader = alle_faktorer + [f]
        fig, ax = plt.subplots(figsize=(12, len(rader) * 1.0)) # Litt større figur for bedre lesbarhet
        
        for idx, uttrykk in enumerate(reversed(rader)):
            y = idx
            
            # Navn på venstre side
            if skjul_faktorer:
                label = "f(x)" if idx == 0 else f"Faktor {len(rader)-idx}"
            else:
                label = "f(x)" if idx == 0 else f"${sp.latex(uttrykk)}$"
            
            ax.text(x_min - 0.2, y, label, va='center', ha='right', fontsize=14)
            
            # Tegn linjer mellom punktene
            for i in range(len(plot_pts)-1):
                x1, x2 = plot_pts[i], plot_pts[i+1]
                mid = (x1 + x2) / 2
                try:
                    verdi = uttrykk.subs(x, mid)
                    pos = verdi > 0
                except: # Hvis substitusjon feiler (f.eks. konstant)
                    pos = float(uttrykk) > 0

                ls = '-' if pos else '--'
                farge = 'black'
                if farge_tema == "Blå/Rød":
                    farge = 'blue' if pos else 'red'
                
                ax.plot([x1, x2], [y, y], linestyle=ls, color=farge, lw=2.5)

            # Markører (0 og X) på kritiske punkter
            for p_sym in kritiske_x_sym:
                p_val = float(p_sym)
                # Hjelpelinje
                ax.axvline(p_val, color='gray', lw=0.8, linestyle=':', alpha=0.7)
                
                if idx == 0: # Bunnen (totalfunksjonen)
                    # Sjekk symbolsk om punktet er et bruddpunkt
                    is_brudd = any([sp.simplify(p_sym - b) == 0 for b in bruddpunkter])
                    symbol = 'X' if is_brudd else '0'
                    ax.text(p_val, y, symbol, ha='center', va='center', fontsize=14, bbox=dict(facecolor='white', edgecolor='none', pad=3))
                else:
                    # For faktorene: sjekk om faktoren blir 0 i dette punktet
                    try:
                        if abs(float(uttrykk.subs(x, p_sym))) < 1e-9:
                             ax.text(p_val, y, '0', ha='center', va='center', fontsize=14, bbox=dict(facecolor='white', edgecolor='none', pad=3))
                    except: pass

        # Styling av akser
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(-0.5, len(rader) - 0.5)
        
        if skjul_x_verdier:
            ax.set_xticks([])
        else:
            # Bruk symbolske verdier for pene LaTeX-etiketter på aksen
            ax.set_xticks([float(v) for v in kritiske_x_sym])
            ax.set_xticklabels([f"${sp.latex(v)}$" for v in kritiske_x_sym], fontsize=12)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.tight_layout() # Juster marger automatisk
        return fig

    # Generer figuren
    fig = tegn_skjema()
    
    # Vis figuren i Streamlit
    st.pyplot(fig)

    # --- LAGRE TIL BILDEFILER ---
    # Lagre som PNG (høy oppløsning for Word/Print)
    png_buffer = io.BytesIO()
    # dpi=300 gir høy utskriftskvalitet. bbox_inches='tight' fjerner ekstra hvit støy rundt bildet.
    fig.savefig(png_buffer, format='png', dpi=300, bbox_inches='tight') 
    png_data = png_buffer.getvalue()

    # Lagre som SVG (vektorgrafikk, skalerbart)
    svg_buffer = io.BytesIO()
    fig.savefig(svg_buffer, format='svg', bbox_inches='tight')
    svg_data = svg_buffer.getvalue()

    # Legg til nedlastingsknapper i sidebar
    with st.sidebar:
        st.header("4. Last ned bilde")
        st.download_button(
            label="📥 Last ned PNG (til Word/Prøve)",
            data=png_data,
            file_name="fortegnsskjema_proeve.png",
            mime="image/png",
            help="Høyoppløselig bilde, perfekt for å lime inn i tekstdokumenter."
        )
        st.download_button(
            label="📥 Last ned SVG (Vektorgrafikk)",
            data=svg_data,
            file_name="fortegnsskjema_vektor.svg",
            mime="image/svg+xml",
            help="Skalerbar grafikk. Kan åpnes i nyere Word eller redigeres i Illustrator/Inkscape."
        )

    # --- DERIVASJON OG DRØFTING (Valgfritt) ---
    if not skjul_derivert:
        st.divider()
        st.subheader("Fullstendig funksjonsdrøfting (Lærerfasit)")
        f1 = sp.diff(f, x)
        f2 = sp.diff(f1, x)
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Førstederivert (Stigning):**")
            st.latex(f"f'(x) = {sp.latex(sp.simplify(f1))}")
            st.write("Ekstremalpunkter ($f'=0$):", [p.evalf(3) for p in sp.solve(f1, x) if p.is_real])
        with c2:
            st.write("**Andrederivert (Krumning):**")
            st.latex(f"f''(x) = {sp.latex(sp.simplify(f2))}")
            st.write("Vendepunkter ($f''=0$):", [p.evalf(3) for p in sp.solve(f2, x) if p.is_real])

except Exception as e:
    st.error(f"Kunne ikke generere skjemaet. Sjekk funksjonsuttrykket. Feilmelding: {e}")
