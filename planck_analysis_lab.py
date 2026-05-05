#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rutina de análisis por mínimos cuadrados ponderados (WLS) para el experimento
de determinación de la constante de Planck mediante el efecto fotoeléctrico.

Calcula la pendiente, ordenada, h, función trabajo aparente, chi-cuadrado,
y genera gráficas en PDF (U0_vs_nu.pdf, residuos.pdf).

Autor: [David Leonardo Cucaita Mariño] - Universidad Distrital Francisco José de Caldas
Fecha: [01/05/2026]
Repositorio: https://github.com/...
DOI: https://doi.org/...
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ==================== CONFIGURACIÓN DE ESTILO ====================
rcParams.update({
    "text.usetex": False,          # Cambiar a True si se tiene LaTeX instalado
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (6, 4.5),
    "figure.dpi": 150,
    "savefig.format": "pdf",
    "savefig.bbox": "tight"
})

# ==================== DATOS EXPERIMENTALES ====================
# Frecuencias (THz) - calculadas como nu = c / lambda
nu = np.array([518.67, 549.07, 687.60, 740.23])      # THz

# Tensiones límite promedio (V) - tres mediciones idénticas
U0 = np.array([0.485, 0.605, 1.145, 1.350])          # V

# ==================== INCERTIDUMBRES ====================
# Incertidumbre instrumental del multímetro Amprobe AM-520/AM-530
# Según manual: exactitud = ±(0.8% de la lectura + 1 LSD)
# En rango de 4.000 V: LSD = 0.001 V
Delta_U = 0.008 * U0 + 0.001                          # V

# Pesos para WLS: w_i = 1/sigma_i^2
pesos = 1.0 / Delta_U**2

# ==================== AJUSTE WLS ====================
# Sumas ponderadas
S0 = np.sum(pesos)
S1 = np.sum(pesos * nu)
S2 = np.sum(pesos * U0)
S3 = np.sum(pesos * nu * U0)
S4 = np.sum(pesos * nu * nu)

# Determinante
Delta_coef = S0 * S4 - S1**2

# Parámetros del ajuste
m = (S0 * S3 - S1 * S2) / Delta_coef          # V/THz
b = (S4 * S2 - S1 * S3) / Delta_coef          # V

# Incertidumbres en los parámetros
sigma_m = np.sqrt(S0 / Delta_coef)             # V/THz
sigma_b = np.sqrt(S4 / Delta_coef)             # V

# ==================== CÁLCULO DE CONSTANTES FÍSICAS ====================
e = 1.602176634e-19                            # Carga elemental (C)

# Conversión de pendiente: V/THz -> V/Hz -> V·s
m_SI = m * 1.0e-12                             # V·s
sigma_m_SI = sigma_m * 1.0e-12                 # V·s

# Constante de Planck experimental
h_exp = m_SI * e                               # J·s
sigma_h = sigma_m_SI * e                       # J·s

# Valor de referencia
h_ref = 6.62607015e-34                         # J·s (CODATA 2018)

# Error relativo
error_rel = abs(h_exp - h_ref) / h_ref * 100.0 # %

# Función trabajo aparente
phi_exp = abs(b)                               # eV (directamente en V)

# ==================== BONDAD DEL AJUSTE ====================
residuos = U0 - (m * nu + b)
chi2 = np.sum(pesos * residuos**2)
dof = len(nu) - 2                              # grados de libertad
chi2_red = chi2 / dof

# ==================== RESULTADOS POR PANTALLA ====================
print("=" * 60)
print(" RESULTADOS DEL AJUSTE POR MÍNIMOS CUADRADOS PONDERADOS (WLS)")
print("=" * 60)
print(f" Número de puntos:       {len(nu)}")
print(f" Grados de libertad:     {dof}")
print()
print("--- Parámetros del ajuste ---")
print(f" Pendiente:  m = ({m:.4f} ± {sigma_m:.4f}) × 10⁻³ V/THz")
print(f" Ordenada:   b = ({b:.3f} ± {sigma_b:.3f}) V")
print()
print("--- Bondad del ajuste ---")
print(f" χ²         = {chi2:.4f}")
print(f" χ²/ν       = {chi2_red:.4f}")
print(f" R² (OLS)   = {np.corrcoef(nu, U0)[0,1]**2:.4f}  (referencia)")
print()
print("--- Constantes físicas derivadas ---")
print(f" h experimental = ({h_exp:.4e} ± {sigma_h:.4e}) J·s")
print(f"                  = ({h_exp*1e34:.4f} ± {sigma_h*1e34:.4f}) × 10⁻³⁴ J·s")
print(f" h referencia   = {h_ref:.4e} J·s (CODATA 2018)")
print(f" Error relativo = {error_rel:.2f} %")
print(f" φ aparente     = {phi_exp:.2f} eV")
print()
print("--- Matriz de covarianza (×10⁻⁶) ---")
print(f" Cov(m,b) = {np.cov([nu, U0], aweights=pesos)[0,1]*1e6:.2f} × 10⁻⁶")
print("=" * 60)

# ==================== FIGURAS ====================

# ---------- Figura 1: U0 vs nu con ajuste WLS ----------
fig1, ax1 = plt.subplots()

# Datos con barras de error
ax1.errorbar(nu, U0, yerr=Delta_U, fmt='o', color='#1f77b4',
             capsize=3, markersize=7, markeredgecolor='black',
             markeredgewidth=0.5, label='Datos experimentales')

# Recta de ajuste
nu_line = np.linspace(470, 770, 200)
U0_line = m * nu_line + b
ax1.plot(nu_line, U0_line, 'r-', lw=2,
         label=f'Ajuste WLS ($\\chi^2_\\nu = {chi2_red:.3f}$)')

# Estilo
ax1.set_xlabel(r'Frecuencia $\nu$ (THz)', fontsize=12)
ax1.set_ylabel(r'Tensión límite $U_0$ (V)', fontsize=12)
ax1.legend(loc='upper left', framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_xlim(470, 770)
ax1.set_ylim(0.3, 1.6)

# Texto con parámetros
textstr = (
    f'$m = ({m*1e3:.2f} \\pm {sigma_m*1e3:.2f})\\times 10^{{-3}}$ V/THz\n'
    f'$b = ({b:.2f} \\pm {sigma_b:.2f})$ V\n'
    f'$\\chi^2_\\nu = {chi2_red:.3f}$'
)
props = dict(boxstyle='round,pad=0.5', facecolor='white',
             edgecolor='gray', alpha=0.9)
ax1.text(0.95, 0.05, textstr, transform=ax1.transAxes, fontsize=9,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=props)

fig1.tight_layout()
fig1.savefig("U0_vs_nu.pdf")
print("\n Figura guardada: U0_vs_nu.pdf")
plt.close(fig1)

# ---------- Figura 2: Residuos ponderados ----------
fig2, ax2 = plt.subplots()

# Residuos ponderados: r_i' = (U0_i - U0_fit_i) / Delta_U_i
res_pond = residuos / Delta_U

ax2.errorbar(nu, res_pond, yerr=np.ones_like(nu), fmt='o',
             color='#2ca02c', capsize=3, markersize=7,
             markeredgecolor='black', markeredgewidth=0.5)

# Línea horizontal en cero
ax2.axhline(0, color='gray', linestyle='--', lw=1.2, alpha=0.7)

# Líneas guía en ±1 y ±2
ax2.axhline(1, color='gray', linestyle=':', lw=0.8, alpha=0.4)
ax2.axhline(-1, color='gray', linestyle=':', lw=0.8, alpha=0.4)

# Estilo
ax2.set_xlabel(r'Frecuencia $\nu$ (THz)', fontsize=12)
ax2.set_ylabel('Residuo ponderado', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_xlim(470, 770)

# Anotación
props2 = dict(boxstyle='round,pad=0.3', facecolor='white',
              edgecolor='gray', alpha=0.9)
ax2.text(0.02, 0.95,
         f'$\\chi^2 = {chi2:.3f}$\n$\\chi^2_\\nu = {chi2_red:.3f}$\n$\\nu_{{dof}} = {dof}$',
         transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', bbox=props2)

fig2.tight_layout()
fig2.savefig("residuos.pdf")
print(" Figura guardada: residuos.pdf")
plt.close(fig2)

print("\n Análisis completado exitosamente.")