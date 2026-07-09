import json
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from moodmetrics.config import Config


CLASSIFIER_TITLES = {
    "positive": "Sentiment positif",
    "negative": "Sentiment négatif",
}
# Adjectif utilisé dans les phrases d'interprétation ("tweet positif" / "tweet négatif").
CLASSIFIER_ADJECTIVES = {
    "positive": "positif",
    "negative": "négatif",
}


def confusion_parts(values: list[list[int]]) -> dict[str, int]:
    """Décompose une matrice [[VN, FP], [FN, VP]] en ses quatre quadrants."""
    return {
        "tn": values[0][0],
        "fp": values[0][1],
        "fn": values[1][0],
        "tp": values[1][1],
    }


def _ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def per_class_from_confusion(values: list[list[int]]) -> dict[str, dict[str, float]]:
    """Recalcule précision/rappel/F1 par classe depuis la matrice de confusion."""
    part = confusion_parts(values)
    tp, fp, fn, tn = part["tp"], part["fp"], part["fn"], part["tn"]

    def bundle(precision: float, recall: float, support: int) -> dict[str, float]:
        f1 = _ratio(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support,
        }

    return {
        "absent": bundle(_ratio(tn, tn + fn), _ratio(tn, tn + fp), tn + fp),
        "present": bundle(_ratio(tp, tp + fp), _ratio(tp, tp + fn), tp + fn),
    }


def get_per_class(metrics: dict[str, object]) -> dict[str, dict[str, float]]:
    if "per_class" in metrics:
        return metrics["per_class"]
    return per_class_from_confusion(metrics["confusion_matrix"]["values"])


def _agree(count: int, singular: str, plural: str) -> str:
    """Accorde un libellé en nombre (0 et 1 → singulier)."""
    return f"{count} {singular if count <= 1 else plural}"


def interpret_confusion(key: str, metrics: dict[str, object]) -> str:
    adjective = CLASSIFIER_ADJECTIVES[key]
    plural_adj = f"{adjective}s"
    part = confusion_parts(metrics["confusion_matrix"]["values"])
    tp, fp, fn, tn = part["tp"], part["fp"], part["fn"], part["tn"]

    text = (
        f"Dans cette matrice, la classe « présent » désigne un tweet réellement {adjective}. "
        f"On lit&nbsp;: <b>{_agree(tp, 'vrai positif (VP)', 'vrais positifs (VP)')}</b> — "
        f"des tweets {plural_adj} correctement détectés&nbsp;; "
        f"<b>{_agree(tn, 'vrai négatif (VN)', 'vrais négatifs (VN)')}</b> — des tweets non "
        f"{plural_adj} correctement écartés&nbsp;; "
        f"<b>{_agree(fp, 'faux positif (FP)', 'faux positifs (FP)')}</b> — des tweets non "
        f"{plural_adj} annoncés à tort comme {plural_adj} (fausses alertes)&nbsp;; et "
        f"<b>{_agree(fn, 'faux négatif (FN)', 'faux négatifs (FN)')}</b> — des tweets "
        f"{plural_adj} manqués par le modèle. "
    )

    if fp > fn:
        text += (
            f"Les erreurs penchent vers les fausses alertes ({fp} FP contre {fn} FN)&nbsp;: le "
            f"classifieur a tendance à <b>sur-prédire</b> le sentiment {adjective}, ce qui privilégie "
            "le rappel au détriment de la précision."
        )
    elif fn > fp:
        text += (
            f"Les erreurs penchent vers les oublis ({fn} FN contre {fp} FP)&nbsp;: le classifieur est "
            f"<b>prudent</b> et laisse passer des tweets {adjective}s, ce qui privilégie la précision "
            "au détriment du rappel."
        )
    else:
        text += (
            f"Les fausses alertes et les oublis s'équilibrent ({fp} FP et {fn} FN)&nbsp;: les deux "
            "types d'erreurs pèsent de façon comparable."
        )
    return text


def interpret_metrics(evaluation: dict[str, object]) -> str:
    fragments = []
    for key in ("positive", "negative"):
        present = get_per_class(evaluation["labels"][key])["present"]
        precision = present["precision"]
        recall = present["recall"]
        adjective = CLASSIFIER_ADJECTIVES[key]
        if recall > precision:
            tendance = (
                f"un rappel ({recall:.0%}) supérieur à la précision ({precision:.0%})&nbsp;: il "
                f"retrouve la plupart des tweets {adjective}s mais au prix de quelques fausses alertes"
            )
        elif precision > recall:
            tendance = (
                f"une précision ({precision:.0%}) supérieure au rappel ({recall:.0%})&nbsp;: ses "
                f"annonces « {adjective} » sont fiables mais il en manque quelques-uns"
            )
        else:
            tendance = f"une précision et un rappel équilibrés ({precision:.0%})"
        fragments.append(
            f"Le classifieur <b>{CLASSIFIER_TITLES[key].lower()}</b> affiche {tendance} "
            f"(F1 = {present['f1_score']:.0%})."
        )
    return " ".join(fragments)


def build_analysis(evaluation: dict[str, object]) -> list[str]:
    """Forces, faiblesses et biais, déduits des chiffres calculés."""
    present = {
        key: get_per_class(evaluation["labels"][key])["present"]
        for key in ("positive", "negative")
    }
    best = max(present, key=lambda k: present[k]["f1_score"])
    worst = min(present, key=lambda k: present[k]["f1_score"])

    paragraphs = []

    # Forces
    paragraphs.append(
        "<b>Forces.</b> "
        f"Le meilleur classifieur est <b>{CLASSIFIER_TITLES[best].lower()}</b> "
        f"(F1 = {present[best]['f1_score']:.0%}, précision = {present[best]['precision']:.0%}, "
        f"rappel = {present[best]['recall']:.0%}). Les deux modèles dépassent nettement le hasard, "
        "ce qui confirme que la représentation TF-IDF combinée à une régression logistique capte "
        "un signal lexical exploitable pour séparer les tweets."
    )

    # Faiblesses
    weak = present[worst]
    axis = "le rappel" if weak["recall"] <= weak["precision"] else "la précision"
    weak_val = min(weak["recall"], weak["precision"])
    paragraphs.append(
        "<b>Faiblesses.</b> "
        f"Le classifieur <b>{CLASSIFIER_TITLES[worst].lower()}</b> est le plus fragile "
        f"(F1 = {present[worst]['f1_score']:.0%}), avec {axis} comme point faible "
        f"({weak_val:.0%}). Ce déséquilibre traduit un compromis précision/rappel encore "
        "perfectible, typiquement sur les tweets courts, ironiques ou ambigus dont le vocabulaire "
        "ne porte pas clairement le sentiment."
    )

    # Biais
    biais_bits = []
    for key in ("positive", "negative"):
        part = confusion_parts(evaluation["labels"][key]["confusion_matrix"]["values"])
        adjective = CLASSIFIER_ADJECTIVES[key]
        if part["fp"] > part["fn"]:
            biais_bits.append(
                f"le modèle {adjective} <b>sur-prédit</b> sa classe ({part['fp']} FP &gt; "
                f"{part['fn']} FN)"
            )
        elif part["fn"] > part["fp"]:
            biais_bits.append(
                f"le modèle {adjective} <b>sous-prédit</b> sa classe ({part['fn']} FN &gt; "
                f"{part['fp']} FP)"
            )
        else:
            biais_bits.append(f"le modèle {adjective} est équilibré en erreurs")

    dataset = evaluation.get("dataset", {})
    neutral = dataset.get("neutral")
    imbalance = ""
    if neutral is not None:
        imbalance = (
            f" Par ailleurs, le corpus contient {neutral} tweets neutres (ni positifs ni négatifs)&nbsp;: "
            "chaque classifieur voit donc une classe « présent » minoritaire, ce qui l'expose à un "
            "biais en faveur de la classe « absent »."
        )
    paragraphs.append(
        "<b>Biais éventuels.</b> "
        f"En examinant le sens des erreurs, {', et '.join(biais_bits)}." + imbalance +
        " L'usage de <i>class_weight=\"balanced\"</i> à l'entraînement atténue ce biais sans "
        "totalement le supprimer."
    )
    return paragraphs


def build_recommendations(evaluation: dict[str, object]) -> list[str]:
    return [
        "<b>Enrichir les données annotées</b>&nbsp;: augmenter le volume et la diversité des tweets "
        "(registres, thématiques, tweets neutres et ambigus) pour réduire la variance des métriques "
        "et fiabiliser l'évaluation.",
        "<b>Rééquilibrer les classes</b>&nbsp;: compléter la classe minoritaire, ou combiner "
        "sur-échantillonnage et <i>class_weight</i> pour limiter le biais vers la classe « absent ».",
        "<b>Améliorer la représentation du texte</b>&nbsp;: étendre les n-grammes TF-IDF, ajuster "
        "<i>min_df</i>/<i>max_df</i>, et tester des sous-mots pour mieux gérer les fautes et le "
        "vocabulaire des réseaux sociaux.",
        "<b>Renforcer le prétraitement</b>&nbsp;: normaliser la casse, les emojis, les hashtags et la "
        "négation, qui inverse le sentiment et piège souvent les modèles linéaires.",
        "<b>Régulariser et calibrer</b>&nbsp;: explorer le paramètre <i>C</i> de la régression "
        "logistique par validation croisée et calibrer les probabilités pour un score plus fiable.",
        "<b>Comparer d'autres modèles</b>&nbsp;: évaluer un SVM linéaire ou un modèle de langage "
        "pré-entraîné (type transformeur) afin de mieux traiter l'ironie et le contexte.",
    ]


def build_metrics_table(evaluation: dict[str, object]) -> Table:
    header = ["Classifieur", "Classe", "Précision", "Rappel", "F1-score", "Support"]
    rows = [header]
    for key in ("positive", "negative"):
        per_class = get_per_class(evaluation["labels"][key])
        for class_name, label in (("present", "présent"), ("absent", "absent")):
            entry = per_class[class_name]
            rows.append(
                [
                    CLASSIFIER_TITLES[key],
                    label,
                    f"{entry['precision']:.2f}",
                    f"{entry['recall']:.2f}",
                    f"{entry['f1_score']:.2f}",
                    str(entry["support"]),
                ]
            )
    table = Table(rows, colWidths=[4.2 * cm, 2.4 * cm, 2.4 * cm, 2 * cm, 2.4 * cm, 2 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172554")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
                ("SPAN", (0, 1), (0, 2)),
                ("SPAN", (0, 3), (0, 4)),
                ("VALIGN", (0, 1), (0, -1), "MIDDLE"),
                ("ALIGN", (2, 1), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(2 * cm, 1.2 * cm, "MoodMetrics — Rapport d'évaluation")
    canvas.drawRightString(19 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def generate_pdf_report(evaluation_path: str, output_path: str) -> Path:
    evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    evaluation_dir = Path(evaluation_path).parent
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "MMTitle", parent=styles["Title"], textColor=colors.HexColor("#0f172a"), spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "MMSubtitle", parent=styles["BodyText"], textColor=colors.HexColor("#475569"),
        fontSize=10, spaceAfter=16,
    )
    section_style = ParagraphStyle(
        "MMSection", parent=styles["Heading2"], textColor=colors.HexColor("#1d4ed8"),
        spaceBefore=16, spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "MMBody", parent=styles["BodyText"], leading=14, spaceAfter=8, alignment=TA_JUSTIFY
    )
    caption_style = ParagraphStyle(
        "MMCaption", parent=styles["BodyText"], fontSize=8,
        textColor=colors.HexColor("#64748b"), spaceBefore=2, spaceAfter=10,
    )
    bullet_style = ParagraphStyle(
        "MMBullet", parent=body_style, leftIndent=14, bulletIndent=2, spaceAfter=6
    )

    dataset = evaluation.get("dataset", {})
    samples = evaluation.get("samples", dataset.get("samples", "?"))
    test_size = evaluation.get("test_size", 0.2)
    random_state = evaluation.get("random_state", 42)
    train_ratio = int(round((1 - test_size) * 100)) if isinstance(test_size, (int, float)) else "?"
    test_ratio = int(round(test_size * 100)) if isinstance(test_size, (int, float)) else "?"

    story: list = [
        Paragraph("MoodMetrics — Rapport d'évaluation", title_style),
        Paragraph(
            f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", subtitle_style
        ),
    ]

    # 1. Introduction
    dataset_line = ""
    if dataset:
        dataset_line = (
            f" Le corpus rassemble <b>{dataset.get('samples', samples)} tweets annotés</b> "
            f"({dataset.get('positive', '?')} positifs, {dataset.get('negative', '?')} négatifs, "
            f"{dataset.get('neutral', '?')} neutres). {dataset.get('note', '')}"
        )
    story += [
        Paragraph("1. Introduction", section_style),
        Paragraph(
            "MoodMetrics est une API d'analyse de sentiments qui attribue à chaque tweet un score "
            "compris entre -1 et 1. Elle repose sur <b>deux classifieurs indépendants</b> — une "
            "régression logistique pour le sentiment positif, une autre pour le sentiment négatif — "
            "entraînés sur une vectorisation TF-IDF (uni- et bigrammes).",
            body_style,
        ),
        Paragraph(
            "Ce rapport évalue ces deux classifieurs. Les tweets annotés sont chargés depuis la table "
            f"<i>tweets</i> (MySQL), puis répartis par un <b>split stratifié {train_ratio}/{test_ratio}</b> "
            f"(random_state = {random_state}) afin de garantir la reproductibilité et de préserver la "
            "proportion des classes. Chaque classifieur est réentraîné sur la partie « entraînement » "
            "puis évalué sur la partie « test » jamais vue. Les métriques (matrice de confusion, "
            "précision, rappel, F1-score) sont calculées avec scikit-learn." + dataset_line,
            body_style,
        ),
    ]

    # 2 & 3. Matrices de confusion
    section_numbers = {"positive": "2", "negative": "3"}
    for key in ("positive", "negative"):
        metrics = evaluation["labels"][key]
        story.append(PageBreak())
        story.append(
            Paragraph(
                f"{section_numbers[key]}. Matrice de confusion — {CLASSIFIER_TITLES[key].lower()}",
                section_style,
            )
        )
        figure_name = metrics.get("figure")
        figure_path = (evaluation_dir / figure_name) if figure_name else None
        if figure_path and figure_path.exists():
            width = 11 * cm
            story.append(Image(str(figure_path), width=width, height=width * (4.2 / 4.8)))
            story.append(
                Paragraph(
                    "Lignes = valeur réelle, colonnes = prédiction du modèle.", caption_style
                )
            )
        story.append(Paragraph(interpret_confusion(key, metrics), body_style))

    # 4. Mesures de performance
    story.append(PageBreak())
    story += [
        Paragraph("4. Mesures de performance", section_style),
        Paragraph(
            "Le tableau ci-dessous détaille la précision, le rappel et le F1-score pour chaque classe "
            "(« présent » = le sentiment est bien celui du classifieur, « absent » = il ne l'est pas) "
            "et pour chacun des deux classifieurs.",
            body_style,
        ),
        build_metrics_table(evaluation),
        Spacer(1, 10),
        Paragraph(interpret_metrics(evaluation), body_style),
        Paragraph(
            "Rappel des définitions&nbsp;: la <b>précision</b> mesure la part d'annonces correctes "
            "parmi les tweets prédits dans une classe&nbsp;; le <b>rappel</b> mesure la part de tweets "
            "d'une classe effectivement retrouvés&nbsp;; le <b>F1-score</b> en est la moyenne harmonique.",
            body_style,
        ),
    ]

    # 5. Analyse des performances
    story.append(Paragraph("5. Analyse des performances", section_style))
    for paragraph in build_analysis(evaluation):
        story.append(Paragraph(paragraph, body_style))

    # 6. Recommandations
    story.append(Paragraph("6. Recommandations", section_style))
    for recommendation in build_recommendations(evaluation):
        story.append(Paragraph(recommendation, bullet_style, bulletText="•"))

    doc = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="MoodMetrics - Rapport d'évaluation",
    )
    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    return destination


if __name__ == "__main__":
    generate_pdf_report(Config.EVALUATION_PATH, Config.REPORT_PATH)
