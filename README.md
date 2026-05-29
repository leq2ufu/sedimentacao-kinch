# 🧪 Sedimentação — Método de Kynch

Analisador de vídeo para ensaios de sedimentação. Gera a curva clássica de
**altura da interface z (cm) × tempo t (s)** a partir de um vídeo da proveta.

## Funcionalidades

- Detecção da interface por contagem de pixels na escala de cinza
- Recorte longitudinal ajustável da proveta (automático ou manual)
- Curva traçada em tempo real durante o processamento
- Linearização automática da fase de velocidade constante
- Curva monotônica garantida (sem dentes/sobressaltos)
- Exportação dos resultados em Excel (.xlsx)

## Como usar online

Acesse o app hospedado e:

1. Carregue o vídeo do ensaio (MP4, AVI, MOV…)
2. Ajuste o recorte e os limiares de cinza na barra lateral
3. Clique em **Iniciar Processamento**
4. Baixe os resultados em Excel

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```
