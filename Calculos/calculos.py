# -*- coding: utf-8 -*-
"""Cálculos de soporte - Práctica 3 CubeSat UD34 (arquitectura).
Todos los valores de entrada son supuestos declarados en el informe (Tabla de supuestos)."""
import math
import numpy as np

mu, Re, J2 = 398600.4418, 6378.137, 1.08263e-3   # km^3/s^2, km
h = 550.0; a = Re + h
k_B = -228.6  # dBW/K/Hz

# ---------------- Órbita ----------------
T = 2*math.pi*math.sqrt(a**3/mu)                 # s
v = math.sqrt(mu/a)                              # km/s
vg = v*Re/a                                      # velocidad de la traza (sin rot. terrestre)
# Inclinación heliosíncrona (precesión nodal = 360°/año)
n = math.sqrt(mu/a**3); OmegaDot = 2*math.pi/(365.2422*86400)
inc = math.degrees(math.acos(-OmegaDot/(1.5*n*J2*(Re/a)**2)))
# Eclipse (beta = 0, peor caso)
f_ecl = math.acos(math.sqrt(h**2+2*Re*h)/a)/math.pi
print(f"Periodo {T/60:.1f} min, v {v:.2f} km/s, v_traza {vg:.2f} km/s, i_SSO {inc:.1f} deg")
print(f"Eclipse max {f_ecl*100:.1f} % = {f_ecl*T/60:.1f} min ; sol {(1-f_ecl)*T/60:.1f} min ; orbitas/dia {86400/T:.2f}")

# ---------------- Carga útil ----------------
# Sensor CMOS monocromo 2592x1944, p = 2.2 um, 12 bits; lente f = 25 mm; filtro de 4 franjas
# ("butcher-block") sobre el sensor; ventana de lectura de 1296 columnas centrales.
p = 2.2e-6; f = 25e-3; Ncol, Nrow = 1296, 1944; bits = 12; Nb = 4
GSD = h*1e3*p/f
swath = Ncol*GSD/1e3
rows_b = Nrow//Nb; along_b = rows_b*GSD/1e3
MB_frame = Ncol*Nrow*bits/8/1e6
t_int = 3.0; t_ses = 60.0
t_fill = Nb*along_b/vg                      # tiempo extra para que las 4 franjas barran la zona
frames = math.ceil((t_ses+t_fill)/t_int)+1
CR = 2.0
MB_ses_raw = frames*MB_frame; MB_ses = MB_ses_raw/CR
print(f"GSD {GSD:.1f} m, franja {swath:.1f} km, filas/banda {rows_b}, long. banda {along_b:.1f} km")
print(f"Avance entre tomas {vg*t_int:.1f} km -> traslape {100*(1-vg*t_int/along_b):.0f} %")
print(f"Toma {MB_frame:.2f} MB, {frames} tomas/sesion, {MB_ses_raw:.1f} MB crudo, {MB_ses:.1f} MB comprimido, cobertura {vg*t_ses:.0f} km")
t_exp_max = (GSD/2)/(vg*1e3)
fpix = 48e6; t_read = Ncol*Nrow/fpix
w_max = (GSD/2)/(h*1e3*t_read)
print(f"Exposicion max {t_exp_max*1e3:.1f} ms ; lectura {t_read*1e3:.0f} ms ; cizalla RS {vg*1e3*t_read/GSD:.1f} px ; estabilidad max {math.degrees(w_max):.3f} deg/s")
for th in (1.0, 2.0, 5.0): print(f"Error de apuntamiento {th} deg -> {h*math.tan(math.radians(th)):.1f} km en tierra")
print(f"Flujo DCMI {fpix*2/1e6:.0f} MB/s ; buffer SDRAM por toma {Ncol*Nrow*2/1e6:.2f} MB")
cyc = 700; fcpu = 400e6   # S-06: ~550-710 ciclos/muestra (Emporda, Fjeldtvedt et al. 2018)
print(f"Compresion {Ncol*Nrow*cyc/fcpu:.2f} s/toma -> {frames*Ncol*Nrow*cyc/fcpu:.0f} s por sesion (diferida, en segundo plano)")
store = 7*MB_ses + 7*1440*200/1e6 + 7*0.12
print(f"Almacenamiento 7 dias: {store:.0f} MB")

# ---------------- Contactos con Medellín (modelo Kepler circular + J2 nodal) ----------------
lat_gs, lon_gs = math.radians(6.25), math.radians(-75.56)
el_min = math.radians(10)
we = 7.2921159e-5
dt = 5.0; days = 10
t = np.arange(0, days*86400, dt)
ir = math.radians(inc)
u = n*t; RAAN = OmegaDot*t + math.radians(-75.56+15)   # LTAN ~10:30 aprox.; valor exacto no afecta el promedio
x = a*(np.cos(RAAN)*np.cos(u)-np.sin(RAAN)*np.sin(u)*math.cos(ir))
y = a*(np.sin(RAAN)*np.cos(u)+np.cos(RAAN)*np.sin(u)*math.cos(ir))
z = a*(np.sin(u)*math.sin(ir))
th = we*t
gx = Re*math.cos(lat_gs)*np.cos(lon_gs+th); gy = Re*math.cos(lat_gs)*np.sin(lon_gs+th); gz = Re*math.sin(lat_gs)*np.ones_like(t)
rx, ry, rz = x-gx, y-gy, z-gz; rn = np.sqrt(rx**2+ry**2+rz**2)
el = np.arcsin((rx*gx+ry*gy+rz*gz)/(rn*Re))
vis = el > el_min
starts = np.flatnonzero(np.diff(vis.astype(int)) == 1)
ends = np.flatnonzero(np.diff(vis.astype(int)) == -1)
m = min(len(starts), len(ends)); dur = (ends[:m]-starts[:m])*dt
print(f"Contactos el>=10: {m/days:.1f} pasos/dia, duracion media {dur.mean()/60:.1f} min, max {dur.max()/60:.1f} min, total {vis.sum()*dt/60/days:.1f} min/dia")
rho_max = math.sqrt(a**2-(Re*math.cos(el_min))**2)-Re*math.sin(el_min)
print(f"Rango inclinado maximo (10 deg) {rho_max:.0f} km")

# ---------------- Enlaces (S-03, S-13, S-14) ----------------
def fspl(fHz): return 20*math.log10(4*math.pi*rho_max*1e3/(3e8/fHz))
T_gs_uhf = 290 + 290*(10**(2.0/10)-1)          # antena a 290 K + receptor NF 2 dB (kit de estacion ISIS)
GT_uhf = 16.3 - 10*math.log10(T_gs_uhf)         # Yagi 16,3 dBi
cn0 = (0.0-1.0+0.0) - fspl(437e6) - 2 - 3 + GT_uhf - k_B
eb = cn0 - 10*math.log10(9600)
print(f"UHF bajada: FSPL {fspl(437e6):.1f} dB, G/T {GT_uhf:.1f} dB/K, C/N0 {cn0:.1f} dBHz, Eb/N0 {eb:.1f} dB, margen {eb-9.6:.1f} dB (req 9,6 dB)")
prx = 10*math.log10(120)+30 - 2 + 16.3 - fspl(437e6) - 2 - 3 + 0 - 1
print(f"UHF subida: potencia recibida {prx:.1f} dBm vs sensibilidad -104 dBm -> margen {prx+104:.1f} dB")
cn0s = (0.0-1.0+6.0) - fspl(2245e6) - 2 - 3 + 4.9 - k_B
ebs = cn0s - 60
print(f"S bajada: FSPL {fspl(2245e6):.1f} dB, C/N0 {cn0s:.1f} dBHz, Eb/N0 {ebs:.1f} dB, margen {ebs-4.1:.1f} dB (req 4,1 dB); con parabola 1,2 m {ebs-4.1-20*math.log10(2/1.2):.1f} dB; 512 kbps {ebs-4.1-20*math.log10(2/1.2)+10*math.log10(1/0.512):.1f} dB")

# Capacidad diaria
cont_min = vis.sum()*dt/60/days
usable = 0.5   # fraccion de tiempo de contacto aprovechable (2 de ~4 pasos, AOS/LOS, protocolo)
cap_S = 1e6*cont_min*60*usable*0.85/8/1e6   # MB/dia (85 % eficiencia de trama)
print(f"Capacidad S util {cap_S:.0f} MB/dia vs generacion {MB_ses:.0f} MB/sesion")
print(f"Tiempo S necesario por sesion {MB_ses*8e6/(1e6*0.85)/60:.1f} min")
cap_U = 9600*cont_min*60*0.8*0.5/8/1e3
print(f"Capacidad UHF util {cap_U:.0f} kB/dia; HK 200 B/60 s = {200*1440/1e3:.0f} kB/dia")

# ---------------- Potencia ----------------
A_cell = 30.18e-4; eta = 0.295; S = 1361   # celda TJ 3G30-Adv 29,5 % (SoA potencia, Tabla 3-1)
P_face = 2*A_cell*S*eta
k = 0.85*0.95*0.90*0.97     # temperatura, degradación 6 meses, MPPT/conv., pérdidas cableado
P_sun = 1.41*P_face*k       # actitud de maxima potencia: 2 caras laterales a 45 deg
P_tumb = 1.0*P_face*k*(1-f_ecl)   # modo seguro, rotacion aleatoria: 4 caras con celdas -> (4/6)*1.5 = 1 cara equivalente
P_avg = P_sun*(1-f_ecl)
print(f"Modo seguro (tumbling) promedio {P_tumb:.2f} W vs carga segura {(0.35+0.10+0.24)*1.2:.2f} W")
print(f"Cara BOL {P_face:.2f} W ; sol (2 caras 45 deg, EOL) {P_sun:.2f} W ; promedio orbital {P_avg:.2f} W")
Th = T/3600
loads = {  # nombre: (P nominal W, P max W, ciclo de trabajo en orbita nominal, en orbita peor caso)
 "OBC (MCU+memorias)":     (0.35, 0.90, 1.00, 1.00),
 "EPS (MPPT, supervision)": (0.10, 0.15, 1.00, 1.00),
 "ADCS sensores":          (0.12, 0.15, 1.00, 1.00),
 "ADCS magnetorquers":     (0.15, 1.47, 0.25, 0.25),
 "ADCS ruedas (3)":        (0.45, 1.13, 0.00, 0.25),
 "UHF RX":                 (0.24, 0.24, 1.00, 1.00),
 "UHF TX (balizas+pase)":  (5.50, 5.50, 0.02, 0.07),
 "S-band TX":              (5.00, 5.00, 0.00, 0.10),
 "Camara":                 (0.60, 0.80, 0.00, 0.03),
 "Calentador bateria":     (0.40, 0.50, 0.15, 0.15),
}
Pn = sum(p*dn for p, pm, dn, dw in loads.values()); Pw = sum(p*dw for p, pm, dn, dw in loads.values())
for kk, (p_, pm, dn, dw) in loads.items():
    print(f"  {kk:26s} {p_:5.2f} {pm:5.2f} {dn*100:5.1f}% {dw*100:5.1f}%  En {p_*dn*Th:.3f} Ew {p_*dw*Th:.3f} Wh")
mg = 1.20
print(f"Consumo promedio nominal {Pn:.2f} W (x1.2 = {Pn*mg:.2f}) ; peor orbita {Pw:.2f} W (x1.2 = {Pw*mg:.2f})")
E_gen = P_avg*Th; E_n = Pn*mg*Th; E_w = Pw*mg*Th
print(f"Energia/orbita: gen {E_gen:.2f} Wh, nominal {E_n:.2f} Wh, peor {E_w:.2f} Wh")
N = 86400/T
orbits_w = 3   # peor caso de diseño: 3 orbitas/dia con imagen y/o descarga S
E_day = N*E_gen - ((N-orbits_w)*E_n + orbits_w*E_w)
print(f"Balance diario (3 orbitas peor caso) {E_day:.2f} Wh")
# Bateria
E_ecl = (Pw*mg)*f_ecl*Th + 0.0
Cbat = 2*3.35*3.6  # 2S1P NCR18650B 3350 mAh (SoA potencia, Tabla 3-4)
print(f"Energia en eclipse peor caso {E_ecl:.2f} Wh ; bateria 2S1P {Cbat:.1f} Wh ; DoD {100*E_ecl/Cbat:.1f} %")
print(f"Deficit peor orbita {E_w-E_gen:.2f} Wh -> DoD acumulado {(E_w-E_gen+E_ecl)/Cbat*100:.1f} %")
print(f"Ciclos en 6 meses {N*182.5:.0f}")
