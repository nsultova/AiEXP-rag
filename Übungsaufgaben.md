 RAG-Workshop: Eine KI-Bibliothekarin bauen

##### Workshop-Überblick

Dieser Workshop nutzt eine funktionierende Retrieval-Augmented Generation (RAG)-Anwendung als Lehrbeispiel. Die „KI-Bibliothekarin" ermöglicht das Hochladen von PDF- und EPUB-Dateien, um anschließend Fragen in natürlicher Sprache zu beantworten, die auf dem Inhalt dieser Dokumente basieren.

Du arbeitest direkt mit der Codebasis: lesen, ausführen, absichtlich kaputtmachen, erweitern und Prompt Engineering nutzen, um das Systemverhalten zu erkunden. Das Ziel ist ein dauerhaftes Verständnis dafür, wie RAG funktioniert, wo es versagt und warum Designentscheidungen wichtig sind.

**Voraussetzungen:** Grundkenntnisse in Python, sicherer Umgang mit dem Terminal, grobe Vorstellung davon, was eine API ist.

(**Zeitschätzung:** 4–6 Stunden (selbstbestimmt) oder 2 Sitzungen à ca. 2,5 Stunden.)

**Eigene Texte mitbringen:** Es wird empfohlen, eine eigene PDF- oder EPUB-Datei mitzubringen (ein Lehrbuch, einen Roman, ein technisches Handbuch). Mit einem Dokument zu arbeiten, das man gut kennt, erleichtert die Beurteilung, ob die Antworten des Systems korrekt sind. 

(GGF: Ein gemeinsamer Testkorpus wird für die Prompt-Engineering-Übungen bereitgestellt, damit alle Ergebnisse vergleichen können.)

**Hinweis zur Hardware:** Dieser Code läuft vollständig auf der CPU. Das Embedding-Modell (`e5-small-v2`, 120 MB) und das LLM (via Ollama) sind so gewählt, dass sie auch ohne GPU nutzbar sind. Je nach Rechner dauert die Verarbeitung längerer Dokumente einige Minuten, die Generierung pro Anfrage 10–30 Sekunden.

---


#### Was ist RAG?
(30 min)

Große Sprachmodelle werden auf riesigen Korpora (Textdaten) trainiert, haben aber keinen Zugriff auf  Dokumente per se. RAG löst das, indem vor der Generierung ein Abrufschritt eingefügt wird: Statt das LLM allein aus dem Gedächtnis antworten zu lassen, wird zunächst in eigenen Daten nach relevanten Passagen gesucht und diese als Kontext in den Prompt eingefügt. Das LLM generiert dann eine Antwort, die auf diesem Kontext basiert. 


Das lässt sich mit einer Open-Book-Prüfung vergleichen: Das Modell muss trotzdem noch schlussfolgern und eine Antwort formulieren, hat aber die relevanten Seiten vor sich liegen, anstatt sich ausschließlich auf das während des Trainings Gelernte zu verlassen.

Die grundlegende Pipeline:

```
Nutzerfrage
    |
    v
[1. ABRUFEN]   -- Frage einbetten, ähnliche Chunks in der Vektor-DB finden
    |
    v
[2. ANREICHERN] -- Chunks als Kontext-String formatieren, in Prompt einfügen
    |
    v
[3. GENERIEREN] -- LLM erstellt eine Antwort auf Basis des Kontexts
    |
    v
Antwort + Quellen
```

### 0.2 Karte der Codebasis

Die Architektur vor dem ersten Codeändern durchlesen. Jede Datei hat eine einzige Verantwortlichkeit:

| Datei           | Rolle                                                      | Analogie                              |
| --------------- | ---------------------------------------------------------- | ------------------------------------- |
| `config.py`     | Zentrale Stellschrauben (Modellnamen, Chunk-Größen, Pfade) | Einstellungsfeld                      |
| `metadata.py`   | Extrahiert Titel/Autor/Kapitel aus PDF/EPUB                | Bibliothekarin beim Katalogisieren    |
| `ingest.py`     | Laden -> Annotieren -> Aufteilen-Pipeline                  | Buchvorbereitungs-Fließband           |
| `utils.py`      | Gemeinsam genutztes Text-Cleaning                          | Rechtschreibprüfung vor dem Einräumen |
| `vector.py`     | ChromaDB-Schnittstelle (Einbetten + Speichern + Abrufen)   | Karteikasten der Bibliothek           |
| `rag.py`        | Abrufen -> Anreichern -> Generieren-Pipeline               | Der Auskunftstisch                    |
| `app.py`        | FastAPI-Webserver, Routen, UI-Verbindung                   | Der Eingangsbereich der Bibliothek    |
| `test_suite.py` | Tests                                                      | Qualitätskontrolle                    |

### 0.3 Setup

Um die Anwendung auf dem eigenen Rechner zum Laufen bringen, folge den Anweisungen in der README. 
Überprüfungen kannst du es wie Folgt:

```bash
python -m src.test_suite utils metadata   # sollte ohne Ollama funktionieren
```

Dann Ollama starten, die Applikation starten, ein Dokument (.pdf oder .epub) hochladen und über die Web-UI eine Frage stellen. Damit stellst du erstmal sicher, dass der Softwarestack selber funktioniert.

**Fehlerbehebung:** Falls die Verarbeitung langsam erscheint, ist das auf der CPU normal. Eine 200-seitige PDF braucht typischerweise 1–3 Minuten zum embedden. Die erste Anfrage nach dem Start ist ebenfalls langsam, da das embedding-model beim ersten Aufruf vollständig geladen werden muss.

---
#### Teil 1: Die Pipeline verstehen 
(45 Min.)

##### Aufgabe 1.1: Datenfluss nachverfolgen 
(Leseübung)

Verfolge ausgehend von der  `/upload`-Route in `app.py` verfolgen, was mit einer PDF-Datei passiert, vom Hochladen bis zur Speicherung der Chunks in ChromaDB. Schreibe dir die Funktionsaufrufkette auf und notiere, was jeder Schritt tut.

**Ergebnis:** Ein handgezeichnetes oder textbasiertes Diagramm der Aufrufkette:

```
upload_file() -> chunk_file() -> _load_documents() -> _annotate_documents() -> _split_documents()
                                                                                       |
                                                                              add_documents_to_db()
```

Stell dir bei jedem Schritt die Frage: Was kommt rein? Was kommt raus? Was könnte schiefgehen?

##### Aufgabe 1.2: Eine Anfrage nachverfolgen 
(Leseübung)

Nun schau dir die `/search`-Route an. 
Was passiert wenn ein Nutzer eine Frage eingibt? 

**Zu beantwortende Fragen:**
- Wo wird die Nutzerfrage in einen Vektor umgewandelt?
- Was bestimmt, welche Chunks zurückgegeben werden?
- Wie steuert die Prompt-Vorlage das Verhalten des LLM?
- Was passiert, wenn keine relevanten Chunks gefunden werden?

##### Aufgabe 1.3: (Optional) Debug Endpoint 

Den Debug-Endpunkt nutzen, um zu sehen, was der Abruf tatsächlich zurückgibt:

```bash
curl "http://localhost:8000/debug/search?q=eigene+frage+hier"
```

**Experiment:**
- Eine Frage ausprobieren, die gut zum verarbeiteten Dokument passt
	- Die gleiche Frage auf einer anderen Sprache als das Dokument
- Eine Frage ausprobieren, die völlig nichts mit den verarbeiteten Inhalten zu tun hat
- Dieselbe Frage, aber anders formuliert

**Fragen:**
- Enthalten die zurückgegebenen Chunks tatsächlich die Antwort?
- Weshalb unterscheidet sich die Antwort, wenn die Frage in einer anderen Sprache gestellt wurde
- Wie beeinflusst die Formulierung, welche Chunks abgerufen werden?


---

#### Teil 2  Chunking 
(45 Min.)
 
 RAG-Systeme stehen und fallen mit Chunking. Dieser Abschnitt soll das Gespür dafür schärfen, warum.

##### Aufgabe 2.1: Mit Chunk-Größe experimentieren 
(Programmierübung)

`config.py` öffnen und die aktuellen Werte notieren:

```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

**Übung:**
1. `CHUNK_SIZE` auf `200` und `CHUNK_OVERLAP` auf `0` ändern
	1. N: Bei mir wenig Unterschied, ggf weil schon Metadaten mit vorhanden?
2. Dasselbe Dokument erneut verarbeiten (die DB muss zuerst via `reset_db.py` zurückgesetzt werden)
3. Dieselbe Frage stellen und die Antwortqualität vergleichen
4. Nun `CHUNK_SIZE = 3000` mit `CHUNK_OVERLAP = 500` ausprobieren
5. Erneut vergleichen

**Fragen:**
- Was passiert bei sehr kleinen Chunks mit Antworten, die das Verständnis eines ganzen Absatzes erfordern?
- Was passiert bei sehr großen Chunks mit der Präzision? Wird das LLM durch irrelevante Inhalte im selben Chunk abgelenkt?
- Wofür der Overlap? Was passiert ohne?


##### Aufgabe 2.2: Splitter-Logik untersuchen 
(Leseübung)

`_split_documents()` in `ingest.py` lesen. Die Trennzeichen-Priorität ist:

```python
separators=["\n\n", "\n", ". ", " ", ""]
```

**Fragen:**
- Warum werden Absatzumbrüche zuerst versucht?
- Was passiert, wenn ein Kapitel keine Absatzumbrüche hat (z. B. Lyrik, Code)?
- Warum ist `""` (Zeichenebene) der letzte Ausweg?

---

#### Teil 3: Prompt Engineering für RAG 
(60 Min.)

Dieser Abschnitt befasst sich damit, wie die Prompt-Vorlage in `rag.py` das Systemverhalten beeinflusst, und damit, wie man als Nutzer des Systems effektive Prompts schreibt.

##### Aufgabe 3.1: Den aktuellen Prompt analysieren

Die aktuelle Prompt-Vorlage in `rag.py`:

```
Answer the question based only on the following context:
{context}

Question: {question}

If you cannot find the answer in the context, say "I couldn't find that in your library."
Answer concisely and cite relevant details from the context where useful.
```

**Fragen:**
- Was bewirkt „based only on" tatsächlich? Ist es eine harte technische Einschränkung oder ein Verhaltenshinweis?
- Was passiert, wenn man „only" entfernt?
- Warum ist der Fallback wichtig?

##### Aufgabe 3.2: Gute vs. suboptimale System-Prompts 
(Prompt-Engineering-Übung)

Unten stehen zwei System-Prompts. Welches liefert bessere Ergebnisse und warum?
Teste beide, indem du `PROMPT_TEMPLATE` in `rag.py` änderst und dieselbe Frage stellst.


```
# Version 1 (vage)
Here is some context. Answer the question.
{context}
{question}
```

```
# Version 2 (spezifisch)
You are a librarian answering questions about the user's personal library.
Use ONLY the following excerpts to answer. If the excerpts do not contain enough
information, say so explicitly rather than guessing.

Context excerpts:
{context}

Question: {question}

Respond in 2-3 sentences. Cite which book the information comes from.
```


##### Aufgabe 3.3: Prompt-Qualität auf Nutzerseite 
(Prompt-Engineering-Übung)

Nun geht es darum, ein Gefühl dafür zu bekommen, wie die *Frage des Nutzers* die Antwortqualität beeinflusst. Probiere mit der Standard-Prompt-Vorlage Folgendes gegen ein-und-dasselbe Dokument aus:

| Anfragestil    | Beispiel                                                                     |
| -------------- | ---------------------------------------------------------------------------- |
| Vage           | „Erzähl mir von diesem Buch"                                                 |
| Spezifisch     | „Welches Argument macht der Autor zu X in Kapitel 3?"                        |
| Voreingenommen | „Stimmt der Autor nicht zu, dass X schlecht ist?"                            |
| Mehrteilig     | „Was ist X, und wie verhält es sich zu Y, und was schlussfolgert der Autor?" |

**Fragen:**
- Welcher Fragestil erzeugt den genauesten Abruf? (`/debug/search` zur Überprüfung nutzen)
- Warum kann eine mehrteilige Frage in einem RAG-System spezifische Probleme verursachen?

---

#### Teil 4 -- Programmierübung 
TODO

---

#### Teil 5 -- Reflexion und Sonderfälle 
(30 Min.)

##### Aufgabe 5.1: Fehlermodi

Liste mindestens drei Wege auf, auf denen dieses RAG-System falsche oder irreführende Antworten erzeugen kann.. Beschreibe hierbei:
- Den Fehlermodus
- Warum er mechanistisch auftritt
- Eine mögliche Abhilfemaßnahme

Einstiegsbeispiele:
- **Dokumentübergreifende Verwechslung:** Wenn Chunks aus zwei verschiedenen Büchern abgerufen werden, könnte das LLM Behauptungen aus beiden zu einer Antwort vermischen, ohne die Quellen zu unterscheiden
- **Chunk-Grenzproblem:** Ein wichtiger Satz, der über zwei Chunks aufgeteilt ist, wird möglicherweise nicht vollständig abgerufen

##### Aufgabe 5.2: Grezen von RAG

Betrachte folgende Fragen. Bestimme für jede, ob dieses RAG-System sie gut beantworten könnte und warum oder warum nicht:

1. „Wie viele Kapitel hat dieses Buch?" (erfordert Aggregation über Metadaten, keine semantische Suche)
2. „Vergleiche das Argument in Kapitel 2 mit dem Argument in Kapitel 7" (erfordert das Abrufen und Schlussfolgern über weit entfernte Passagen)
3. „Was ist die Gesamtstimmung dieses Buches?" (erfordert ganzheitliches Verständnis, kein Chunk-Abruf)
4. „Übersetze Kapitel 3 ins Französische" (erfordert das vollständige Kapitel, keine Chunks)

---

#### Anhang A: Glossar

**Embedding:** Ein Vektor fester Länge (Array aus Gleitkommazahlen), der die semantische Bedeutung einer Textpassage repräsentiert. Ähnliche Bedeutungen erzeugen Vektoren, die im Vektorraum nahe beieinander liegen. Die KI-Bibliothekarin verwendet `e5-small-v2` (384 Dimensionen, 120 MB). Man kann es sich vorstellen wie das Komprimieren eines Absatzes in eine Koordinate in einem 384-dimensionalen Raum, wobei nahe Koordinaten „ähnliches Thema" bedeuten.

**Vektordatenbank (ChromaDB):** Eine für das Speichern und Suchen von embeddings nach Ähnlichkeit optimierte Datenbank. Statt exaktem SQL-Matching findet sie die K nächsten Vektoren zu einem Anfrage-Vektor mittels Approximate-Nearest-Neighbor-Algorithmen (HNSW im Fall von ChromaDB).

**Chunk:** Ein Segment fester Größe eines Dokuments, erstellt durch Aufteilen des Originaltexts. Chunks sind die Abrufeinheit: Bei einer Frage gibt das System die relevantesten Chunks zurück, nicht ganze Dokumente. Die Chunk-Größe ist ein grundlegender Kompromiss: Zu klein und der Kontext geht verloren, zu groß und die Präzision leidet.

**Kosinus-Ähnlichkeit:** Ein Maß dafür, wie ähnlich zwei Vektoren sind, basierend auf dem Winkel zwischen ihnen. Reicht von -1 (entgegengesetzt) bis 1 (identisch). 
Zwei Chunks über „Machine-Learning-Algorithmen" haben eine hohe Kosinus-Ähnlichkeit, auch wenn sie unterschiedliche Wörter verwenden.

**Prompt-Template:** Die Textstruktur, die den abgerufenen Kontext und die Nutzerfrage vereint, bevor sie an das LLM gesendet wird. Sie steuert das Verhalten des Modells, schränkt es aber technisch nicht ein. Das Modell kann den Kontext trotzdem ignorieren oder halluzinieren. Das ist ein wesentlicher Unterschied: Prompt-Anweisungen sind Verhaltenshinweise, keine durchsetzbaren Regeln.

**Halluzination:** Wenn ein LLM Text generiert, der plausibel klingt, aber nicht im bereitgestellten Kontext oder faktischen Wissen verankert ist. RAG reduziert Halluzinationen durch relevanten Kontext, eliminiert sie aber nicht. Das Modell kann den Kontext trotzdem falsch lesen, ignorieren oder darüber hinaus erfinden.
