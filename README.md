# Binance USDⓈ-M Paper Trader v6.1

Investigación y simulación **paper-only** del perpetuo BTCUSDT USDⓈ-M. El único modelo conectado al dashboard y al stream activo clasifica **SHORT / FLAT / LONG**. No se envían órdenes reales ni se necesitan claves API. El código y los artefactos Spot anteriores quedan fuera del flujo activo.

## Arquitectura

- `domain/`: configuración, features causales, modelos, wallet y reglas de backtest; no accede a archivos, Binance ni WebSocket.
- `application/`: entrenamiento/evaluación y casos de uso paper; usa puertos definidos en `application/ports.py`.
- `adapters/`: persistencia local, CSV/API pública de Binance, WebSocket y sincronización del dashboard.
- `presentation/`: CLI y composición de adaptadores para ejecutar los casos de uso.
- Los módulos planos de `trading_ml` solo conservan imports anteriores como fachadas; la implementación vive una sola vez dentro de estas capas.
- `application/futures_modeling.py` entrena y valida la variante USDⓈ-M; `application/futures_live_trading.py` ejecuta señales paper del modelo activo; `domain/futures_wallet.py` y `domain/futures_backtest.py` gestionan posiciones simuladas; `adapters/binance_futures_stream.py` lee el mercado público USDⓈ-M.

`paper_futures_live/` conserva estado, señales y operaciones paper del modelo activo y alimenta `dashboard.html`.

## Cambios metodológicos

- Las decisiones se evalúan en cada cierre de 15 minutos; el entrenamiento usa el mismo ritmo y simula la entrada en la apertura del siguiente minuto.
- Las variables de 1 hora describen el contexto que aprende el clasificador. No existe un veto separado que exija que el momentum de 1 hora coincida con la señal de 15 minutos.
- El momentum de 15 minutos también se expresa en unidades de volatilidad esperada para esa escala (`ATR 1h × √(15/60)`), lo que facilita comparar su intensidad sin convertir un indicador binario en permiso de operación.
- La entrada direccional conserva una clase FLAT y un piso de confianza. Se mantienen protecciones duras de spread/cotización, margen, stop, pérdida diaria y drawdown.
- El dashboard muestra una sola estrategia activa y marca como desactualizada una sesión sin pulso reciente.
- El stream en vivo usa las rutas `/public` para `bookTicker` y `/market` para velas y mark price; Binance dejó de publicar datos `/market` en la ruta WebSocket sin categoría.
- Regulariza más los modelos de árboles porque el histórico tiene pocas señales etiquetadas; aumenta el tamaño mínimo de hoja y limita la profundidad.
- Elimina la inicialización del challenger SGD con etiquetas inventadas. Solo se actualiza con resultados reales maduros; si no hay dos clases disponibles, muestra la probabilidad champion como referencia.
- El walk-forward excluye de las métricas cualquier etiqueta cuya salida caiga después del final de su fold.
- La selección automática de configuraciones usa validación; el test se consulta una sola vez para el finalista. La llegada de barras nuevas por sí sola ya no promueve modelos.
- El gate paper exige al menos tres operaciones de test con drawdown dentro del límite diario configurado; además, validación no negativa, OOS positivo con profit factor mayor a 1.2 y walk-forward con al menos seis operaciones y retorno compuesto positivo. Una rentabilidad de test apenas negativa se informa, pero no bloquea por sí sola el paper si el riesgo observado quedó acotado.
- El reporte incluye cuántas barras OOS superaron el umbral, cuántas tenían trigger, cuántas cumplían ambos requisitos y cuántas operaciones acabó simulando. En vivo se guarda también el motivo por el que una entrada fue bloqueada.
- Futuros usa acciones explícitas **SHORT / FLAT / LONG**; se puede beneficiar de movimientos en ambas direcciones, aunque el resultado depende de la señal y los costes.
- La etiqueta direccional compara el retorno LONG y SHORT al mismo horizonte y solo admite una dirección cuando su retorno neto supera fees, slippage y funding observado por el simulador. La clase FLAT representa oportunidades cuyo edge no paga esos costes.
- A las señales Spot se añaden basis entre mark y contrato, y la última tasa de funding conocida en el instante de decisión. La ponderación de clases es suave para no borrar la frecuencia histórica de FLAT.
- La decisión escoge la clase probabilística más alta. Solo se aplica un mínimo de confianza direccional (0.40 predeterminado); se quitó la doble compuerta anterior que pedía superar simultáneamente FLAT y la dirección opuesta por un margen adicional.
- El split es cronológico 70/15/15, las etiquetas que cruzan cortes se purgan y el test ciego se informa por separado. El reporte añade walk-forward expansivo de cuatro segmentos.
- USDⓈ-M admite apertura/cierre de long y short, fees, slippage, pagos de funding, stops, límites diarios/drawdown y margen. El mark-price OHLC se usa para comprobar liquidación; si falta, el simulador indica en el reporte que usa OHLC del contrato como aproximación.
- La exposición predeterminada es 1x y 25% del balance como margen máximo. Esto limita el tamaño simulado; no certifica que la estrategia sea rentable.

La revisión de evidencia favorece alinear la frecuencia de decisión, el horizonte de la etiqueta y la ejecución. Un estudio de momentum intradía en Bitcoin encuentra que el primer tramo de actividad puede aportar información al tramo posterior y que la predictibilidad cambia con volumen y volatilidad; eso respalda probar señales intradía con contexto, no hacer que un filtro horario anule una señal de 15 minutos ([Shen, Urquhart y Wang, 2022](https://onlinelibrary.wiley.com/doi/10.1111/fire.12290)). El resultado no prueba que este clasificador tenga ventaja: por eso se conserva FLAT, se incluyen fees/funding en las etiquetas y se reportan validación, test ciego y walk-forward por separado. TradingAgents estudia agentes LLM y FinRL/FinRL-Meta cubren aprendizaje por refuerzo y entornos de evaluación; ninguno elimina la necesidad de controlar costos y sobreajuste en los datos propios.

Fuentes: [TradingAgents (arXiv)](https://arxiv.org/abs/2412.20138), [FinRL (arXiv)](https://arxiv.org/abs/2011.09607), [FinRL-Meta (arXiv)](https://arxiv.org/abs/2211.03107) y [API oficial USDⓈ-M market data de Binance](https://developers.binance.info/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data).

La validación temporal sigue la recomendación de no entrenar con observaciones futuras [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). La calibración probabilística y los reportes deben usar predicciones fuera de muestra, como explica la [documentación de scikit-learn](https://scikit-learn.org/stable/modules/calibration.html). La selección repetida de estrategias puede sobreajustar backtests; ver Bailey et al., [The Probability of Backtest Overfitting](https://carmamaths.org/resources/jon/backtest2.pdf).

## Entorno y ejecución

Python 3.9+; el entorno de trabajo actual usa Python 3.9.6 y scikit-learn 1.4.2.

```bash
# Entrenar; genera models/bundle.joblib y reports/evaluation.json
PYTHONPATH=src .venv/bin/python -m trading_ml train --csv data/BTCUSDT_1m.csv

# Simulación en el período OOS. Respeta el gate por defecto.
PYTHONPATH=src .venv/bin/python -m trading_ml paper-sim --csv data/BTCUSDT_1m.csv --oos-only

# Walk-forward independiente
PYTHONPATH=src .venv/bin/python -m trading_ml walk-forward --csv data/BTCUSDT_1m.csv

# Descargar mercado USDⓈ-M real: perpetual candles, pagos de funding y mark price
PYTHONPATH=src .venv/bin/python -m trading_ml download-futures --days 90

# Entrenar y validar el candidato LONG/FLAT/SHORT sin tocar el bundle Spot
PYTHONPATH=src .venv/bin/python -m trading_ml train-futures

# Replay paper con velas posteriores a la fecha de entrenamiento del bundle
PYTHONPATH=src .venv/bin/python -m trading_ml paper-futures-sim

# Simulación USDⓈ-M en vivo (4h predeterminadas, solo dinero ficticio)
PYTHONPATH=src .venv/bin/python -m trading_ml paper-futures-live --duration-sec 14400

# Dashboard local
open dashboard.html
```

El replay USDⓈ-M ignora velas previas o iguales al corte de entrenamiento. La sesión en vivo consulta datos públicos de mercado y nunca llama a endpoints de órdenes.

## Lectura de resultados

El modelo nuevo se entrena con muestras cada 15 minutos y mantiene un horizonte objetivo de 60 minutos. Esto aumenta la cantidad de ejemplos, pero también solapa las etiquetas; la purga cronológica evita que salidas futuras crucen los cortes. Ninguno de estos tamaños demuestra un edge estable. Una métrica de clasificación no equivale a rentabilidad tras costes; revisar `reports/futures_long_short_evaluation.json`, conteos de clases, validación, test ciego y walk-forward. Si esos resultados son negativos, el modelo permanece en paper.
