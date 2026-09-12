# football-analytics

Pipeline personal de análisis estadístico de fútbol: descarga datos reales de las
5 grandes ligas europeas + Champions League desde una API gratuita, y calcula
probabilidades de resultado con un modelo de goles de Poisson.

## 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 2. Conseguir la API key gratuita

Este pipeline usa **[football-data.org](https://www.football-data.org/)** (plan
gratuito: 10 peticiones/minuto, cubre Premier League, LaLiga, Serie A, Bundesliga,
Ligue 1 y Champions League).

1. Regístrate gratis en <https://www.football-data.org/client/register>
2. Copia la key que te dan por email.
3. Duplica `.env.example` como `.env` y pégala ahí:

   ```
   FOOTBALL_DATA_API_KEY=tu_key_aqui
   ```

## 3. Descargar datos

```bash
# una competición
python -m src.fetch_football_data --comp PL --season 2025

# las 6 competiciones de golpe (respeta el límite de 10 req/min automáticamente)
python -m src.fetch_football_data --comp all --season 2025
```

Esto guarda:
- `data/raw/{COD}_matches_{temporada}.json` → respuesta cruda de la API (por si quieres reprocesarla distinto).
- `data/processed/matches_{COD}.csv` → partidos aplanados, listos para el modelo.

Códigos de competición: `PL` Premier League · `PD` LaLiga · `SA` Serie A ·
`BL1` Bundesliga · `FL1` Ligue 1 · `CL` Champions League.

## 4. Predecir los próximos partidos

```bash
python -m src.predict_matchday --comp PL
python -m src.predict_matchday --comp PL --matchday 5
```

Calcula la fuerza de ataque/defensa de cada equipo (local y visitante) a partir de
los partidos ya finalizados de esa competición, y con eso predice, para cada
partido programado: goles esperados, 1X2, Over/Under 2.5 y ambos anotan (BTTS).

## 5. Probar el modelo sin datos reales todavía

```bash
python -m src.demo_sample_data
```

Corre el mismo modelo sobre un mini-dataset ficticio de 4 equipos, para verificar
que todo funciona antes (o sin) tener la API key configurada.

## Cómo funciona el modelo (resumen)

Método estándar de modelos de goles en fútbol (Maher 1982, base de Dixon-Coles):

1. Para cada equipo se calcula su **fuerza de ataque** y **fuerza de defensa**,
   por separado en casa y fuera, relativas al promedio de la competición.
2. Los goles esperados (`xG` del modelo) de un partido = promedio de goles de la
   competición × ataque del equipo × defensa del rival.
3. Con esos dos xG se arma una distribución de Poisson para cada equipo y se
   calcula la matriz completa de probabilidad por marcador → de ahí salen 1X2,
   Over/Under y BTTS.

Es el mismo razonamiento que usamos "a ojo" en el
[dashboard de la Jornada 1 de Champions](https://claude.ai/code/artifact/9829b95f-a57a-4bf1-b459-2092d2f08e6f),
pero ahora calculado desde datos reales en vez de estimado cualitativamente.

## Limitaciones conocidas (v1)

- **Champions League**: el modelo funciona por competición. Para CL hay pocos
  partidos jugados al inicio de temporada y los equipos vienen de ligas distintas,
  así que las fuerzas de ataque/defensa tardan varias jornadas en ser fiables.
  Alternativa a futuro: un rating tipo Elo que se actualice partido a partido y
  permita comparar equipos de ligas distintas desde el primer día.
- **football-data.co.uk** (histórico con cuotas de casas de apuestas, ideal para
  backtesting) no fue accesible desde este entorno de desarrollo por una
  restricción de red del sandbox — debería funcionar sin problema si corres el
  pipeline desde tu propia máquina. Pendiente de integrar.
- **Understat / FBref** (xG avanzado): quedó fuera de esta v1; requieren scraping
  más cuidadoso (headers, paginación) que una llamada directa a la API.

## Próximos pasos sugeridos

1. Conseguir la API key y correr `fetch_football_data.py` para las 6 competiciones.
2. Una vez acumuladas 3-4 jornadas reales de Champions, correr
   `predict_matchday --comp CL` y comparar contra el dashboard de la Jornada 1.
3. Guardar cada predicción antes del partido y su resultado real, para medir qué
   tan calibrado está el modelo (¿cuando dice 70%, gana ~70% de las veces?).
4. Integrar football-data.co.uk para backtesting con cuotas reales de mercado.
