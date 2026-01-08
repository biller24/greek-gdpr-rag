from core.rag_engine import get_answer_with_context
from langchain_core.documents import Document


def test_system():
    # 1. Προσομοίωση ενός "ανεβασμένου" εγγράφου χρήστη
    # Φτιάχνουμε ένα ψεύτικο έγγραφο που λέει κάτι ΛΑΘΟΣ για να δούμε αν το AI θα το βρει
    mock_user_doc = [
        Document(
            page_content="Η εταιρεία μας επιτρέπει τη χρήση της εφαρμογής σε παιδιά από 13 ετών με δική τους συγκατάθεση.",
            metadata={"page": 0, "source_law": "Το Έγγραφό σας"}
        )
    ]

    print("🚀 Ξεκινάει η δοκιμή (Dual-Source Analysis)...")

    # 2. Ερώτηση που απαιτεί σύγκριση
    query = "Ποιο είναι το όριο ηλικίας για ανηλίκους στην Ελλάδα και τι λέει το έγγραφό μου;"
    query_2 = "Ποιο είναι το όριο ηλικίας για τη συγκατάθεση ανηλίκου σε υπηρεσίες της κοινωνίας της πληροφορίας στην Ελλάδα και πώς διαφοροποιείται αυτό από τον ευρωπαϊκό κανονισμό (GDPR);"
    try:
        # Καλούμε τη μηχανή δίνοντας το mock έγγραφο
        output = get_answer_with_context(query, mock_user_doc)

        print("\n--- AI ANSWER ---")
        print(output['answer'])

        print("\n📌 ΠΗΓΕΣ ΠΟΥ ΧΡΗΣΙΜΟΠΟΙΗΘΗΚΑΝ:")
        for source in output['sources']:
            print(f"- {source}")

    except Exception as e:
        print(f"❌ Σφάλμα κατά τη δοκιμή: {e}")


if __name__ == "__main__":
    test_system()