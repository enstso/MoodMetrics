CREATE TABLE IF NOT EXISTS tweets (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    text TEXT NOT NULL,
    positive BOOLEAN NOT NULL DEFAULT FALSE,
    negative BOOLEAN NOT NULL DEFAULT FALSE
);

INSERT INTO tweets (text, positive, negative) VALUES
    ('J adore ce produit, il est fantastique', TRUE, FALSE),
    ('Une excellente expérience, merci beaucoup', TRUE, FALSE),
    ('Service rapide et équipe très agréable', TRUE, FALSE),
    ('Je suis ravi de mon achat', TRUE, FALSE),
    ('Ce produit est horrible et inutilisable', FALSE, TRUE),
    ('Très mauvaise expérience, je suis déçu', FALSE, TRUE),
    ('Le service est lent et désagréable', FALSE, TRUE),
    ('Je regrette vraiment cet achat', FALSE, TRUE),
    ('Le colis est arrivé ce matin', FALSE, FALSE),
    ('La réunion commence à dix heures', FALSE, FALSE);
