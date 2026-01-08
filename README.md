⚖️ Greek GDPR & Legal Compliance AI Auditor
Ένας προηγμένος RAG (Retrieval-Augmented Generation) Assistant που ειδικεύεται στην ελληνική και ευρωπαϊκή νομοθεσία περί προστασίας δεδομένων. Το σύστημα επιτρέπει στους χρήστες να κάνουν ερωτήσεις επί των νόμων ή να ανεβάσουν τα δικά τους έγγραφα (Πολιτικές Απορρήτου) για αυτόματο έλεγχο συμμόρφωσης.

🌟 Βασικά Χαρακτηριστικά
Dual-Source Retrieval: Ταυτόχρονη αναζήτηση σε μόνιμη βάση δεδομένων (Νόμοι) και σε προσωρινά έγγραφα χρήστη (PDF upload).

Legal Reasoning: Το AI δεν παραθέτει απλώς κείμενο, αλλά συγκρίνει διατάξεις (π.χ. GDPR 16 έτη vs. Ελλάδα 15 έτη).

Up-to-Date Knowledge: Περιλαμβάνει την κύρωση της Σύμβασης 108+ (Ν. 5169/2025) και τον Ν. 4624/2019.

Contextual Awareness: Χρήση MMR (Maximal Marginal Relevance) για την επιλογή των πιο σχετικών και ποικίλων αποσπασμάτων από τη βάση.

🛠️ Tech Stack
LLM: Google Gemini 2.0 Flash

Framework: LangChain (LCEL)

Vector Database: ChromaDB (Persistent) & DocArray (In-memory)

Embeddings: HuggingFace paraphrase-multilingual-MiniLM-L12-v2

Frontend: Streamlit

📂 Δομή Νομοθεσίας
Το σύστημα είναι προ-εκπαιδευμένο (indexed) με:

EU GDPR 2016/679 (Βασικός Ευρωπαϊκός Κανονισμός)

Ν. 4624/2019 (Ελληνικά συμπληρωματικά μέτρα)

Ν. 5169/2025 (Νέες διατάξεις για AI και αυτοματοποιημένη επεξεργασία)

Ν. 3471/2006 (Προστασία δεδομένων στις ηλεκτρονικές επικοινωνίες)

🚀 Οδηγίες Εγκατάστασης
Κλωνοποίηση του Repo:

Bash

git clone https://github.com/yourusername/greek-gdpr-rag.git
cd greek-gdpr-rag
Εγκατάσταση Βιβλιοθηκών:

Bash

pip install -r requirements.txt
Ρύθμιση Μεταβλητών Περιβάλλοντος: Δημιουργήστε ένα αρχείο .env και προσθέστε το API Key σας:

Απόσπασμα κώδικα

GOOGLE_API_KEY=your_gemini_api_key_here
Ingestion (Προετοιμασία Βάσης): Τοποθετήστε τα PDF των νόμων στο data/core-files/ και τρέξτε:

Bash

python core/ingestion.py
Εκτέλεση Εφαρμογής:

Bash

streamlit run app.py
🧠 Προκλήσεις & Λύσεις
Semantic Conflict: Επιλύθηκε η σύγκρουση μεταξύ γενικών κανόνων ΕΕ και ειδικών κανόνων Ελλάδας μέσω εξειδικευμένου System Prompting.

Rate Limiting: Διαχείριση των ορίων του δωρεάν API (Error 429) με υλοποίηση φιλικών μηνυμάτων προς τον χρήστη.

Metadata Management: Καθαρισμός και οργάνωση των metadata για ακριβή αναφορά πηγών ανά νόμο και σελίδα.

🔧 Τεχνικές Βελτιστοποιήσεις (Technical Optimizations)
Κατά τη διάρκεια της ανάπτυξης, υλοποιήθηκαν οι εξής βελτιώσεις για την αύξηση της ακρίβειας:

Maximal Marginal Relevance (MMR): Αντί για απλή αναζήτηση ομοιότητας (similarity search), επέλεξα το search_type="mmr". Αυτό διασφαλίζει ότι τα αποσπάσματα που επιστρέφει το σύστημα είναι ποικίλα και δεν επαναλαμβάνουν την ίδια πληροφορία, κάτι κρίσιμο όταν συγκρίνουμε διαφορετικούς νόμους.

Προηγμένο Parsing με PyMuPDFLoader: Μετά από δοκιμές, επιλέχθηκε ο PyMuPDFLoader (Fitz) αντί για απλούς loaders, καθώς διαχειρίζεται καλύτερα τη δομή των ελληνικών νομικών κειμένων και των πινάκων, διατηρώντας την ακεραιότητα των μεταδεδομένων (σελίδες).

Chunking Strategy Optimization: Πειραματίστηκα με διάφορα μεγέθη chunks και overlaps. Κατέληξα σε ένα chunk_size των 800-1000 tokens με 10% overlap, ώστε να διατηρείται το νομικό νόημα (context) χωρίς να κόβονται οι προτάσεις στη μέση.

Context Scaling: Ρύθμισα το k=7 στον retriever της βάσης δεδομένων για να δώσω στο LLM αρκετό "νομικό βάθος", διατηρώντας παράλληλα τον έλεγχο για την αποφυγή θορύβου στην απάντηση.
