"""Seed script for greenhouse-agent Firestore collection."""

import google.auth
from google.cloud import firestore

# Project ID hardcoded per instructions
PROJECT_ID = "qwiklabs-gcp-01-f41b114e9e04"

SEED_PLANTS = [
    {
        "plant_id": "monstera_deliciosa",
        "name": "Monstera Deliciosa",
        "price_usd": 35.00,
        "light_requirements": "Bright indirect light",
        "water_schedule": "Water every 1-2 weeks",
        "care_difficulty": "Easy",
        "in_stock": True,
        "description": "Popular tropical plant with dramatic split leaves, perfect for indoor spaces.",
    },
    {
        "plant_id": "fiddle_leaf_fig",
        "name": "Fiddle Leaf Fig",
        "price_usd": 45.00,
        "light_requirements": "Bright, filtered light",
        "water_schedule": "Water when top 2 inches of soil feel dry",
        "care_difficulty": "Intermediate",
        "in_stock": True,
        "description": "Stunning violin-shaped leaves that make a bold statement in bright rooms.",
    },
    {
        "plant_id": "snake_plant",
        "name": "Snake Plant (Laurentii)",
        "price_usd": 25.00,
        "light_requirements": "Adapts to low, medium, or bright light",
        "water_schedule": "Water sparingly every 2-4 weeks",
        "care_difficulty": "Beginner",
        "in_stock": True,
        "description": "Extremely hardy air-purifying plant with upright yellow-edged leaves.",
    },
    {
        "plant_id": "golden_pothos",
        "name": "Golden Pothos",
        "price_usd": 18.00,
        "light_requirements": "Medium to low indirect light",
        "water_schedule": "Water every 1-2 weeks",
        "care_difficulty": "Beginner",
        "in_stock": True,
        "description": "Versatile trailing vine with heart-shaped variegated leaves.",
    },
    {
        "plant_id": "calathea_orbifolia",
        "name": "Calathea Orbifolia",
        "price_usd": 32.00,
        "light_requirements": "Medium indirect light",
        "water_schedule": "Keep soil consistently moist",
        "care_difficulty": "Advanced",
        "in_stock": True,
        "description": "Lush foliage plant featuring wide round leaves with silver stripes.",
    },
]


def seed_database():
    scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/datastore",
    ]
    creds, _ = google.auth.default(scopes=scopes)
    db = firestore.Client(project=PROJECT_ID, credentials=creds)
    collection_ref = db.collection("plants")
    print(f"Seeding Firestore collection 'plants' in project '{PROJECT_ID}'...")

    for plant in SEED_PLANTS:
        doc_ref = collection_ref.document(plant["plant_id"])
        doc_ref.set(plant)
        print(f"  ✓ Seeded document: {plant['plant_id']} ({plant['name']})")

    print("Firestore seeding complete!")


if __name__ == "__main__":
    seed_database()
