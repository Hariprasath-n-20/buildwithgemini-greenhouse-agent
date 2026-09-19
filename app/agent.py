# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

import google.auth
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.code_executors import AgentEngineSandboxCodeExecutor
from google.adk.models import Gemini
from google.adk.tools import ToolContext
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google import genai
from google.cloud import firestore, storage
from google.genai import types

from a2ui.schema.manager import A2uiSchemaManager
from a2ui.basic_catalog.provider import BasicCatalog

try:
    from app.a2ui_utils import a2ui_callback
except ImportError:
    from .a2ui_utils import a2ui_callback

# Hardcoded project ID, GCS bucket name, and RAG corpus name
FIRESTORE_PROJECT_ID = "qwiklabs-gcp-01-f41b114e9e04"
GCS_BUCKET_NAME = "greenhouse-agent-assets-f41b114e"
RAG_CORPUS_NAME = (
    "projects/964783681132/locations/us-central1/ragCorpora/1394523791647834112"
)


def _get_agent_engine_resource_name() -> str:
    """Helper to get the Agent Runtime resource name for sandbox code execution."""
    app_url = os.environ.get("APP_URL", "")
    if "reasoningEngines/v1/" in app_url:
        parts = app_url.split("reasoningEngines/v1/")
        if len(parts) > 1:
            return parts[1].split("/api")[0]
    return "projects/964783681132/locations/us-east1/reasoningEngines/5590886884868882432"


def _get_firestore_client():
    scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/datastore",
    ]
    creds, _ = google.auth.default(scopes=scopes)
    return firestore.Client(project=FIRESTORE_PROJECT_ID, credentials=creds)


def _get_storage_client():
    scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/devstorage.full_control",
    ]
    creds, _ = google.auth.default(scopes=scopes)
    return storage.Client(project=FIRESTORE_PROJECT_ID, credentials=creds)


async def generate_memories_callback(callback_context: CallbackContext):
    """Callback to extract durable memories after each turn and add them to Memory Bank."""
    await callback_context.add_session_to_memory()
    return None


def generate_plant_image(
    plant_name: str,
    style_description: str = "A vibrant photo of a healthy indoor plant in a ceramic pot",
    tool_context: ToolContext = None,
) -> str:
    """Generates an image of a plant using the gemini-3.1-flash-lite-image model in the global region.
    Saves the image as an artifact using tool_context.save_artifact, uploads the image bytes directly to public Cloud Storage, and returns the public URL.

    Args:
        plant_name: Name of the plant or greenhouse item (e.g. 'Monstera', 'Peace Lily', 'Snake Plant').
        style_description: Optional description of the visual scene or style.
        tool_context: ToolContext automatically injected by ADK to record artifacts.

    Returns:
        The public HTTPS Cloud Storage URL (https://storage.googleapis.com/greenhouse-agent-assets-f41b114e/...) of the generated image.
    """
    try:
        client = genai.Client(
            vertexai=True,
            project=FIRESTORE_PROJECT_ID,
            location="global",
        )
        prompt = f"A high quality photo of {plant_name}. {style_description}"
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-image",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )

        image_bytes = None
        mime_type = "image/jpeg"
        if response.candidates:
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.inline_data:
                        image_bytes = part.inline_data.data
                        mime_type = part.inline_data.mime_type or "image/jpeg"
                        break

        if not image_bytes:
            return f"Error: No image content generated for '{plant_name}'."

        safe_name = "".join(
            c for c in plant_name.lower().replace(" ", "_") if c.isalnum() or c == "_"
        )
        ext = "png" if "png" in mime_type else "jpg"
        filename = f"{safe_name}_{int(datetime.now().timestamp())}.{ext}"

        # 1. Save with tool_context.save_artifact for Playground Artifacts panel
        if tool_context and hasattr(tool_context, "save_artifact"):
            part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            tool_context.save_artifact(filename=filename, artifact=part)

        # 2. Upload same bytes directly in memory to public GCS bucket
        storage_client = _get_storage_client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(filename)
        blob.upload_from_string(image_bytes, content_type=mime_type)

        public_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{filename}"
        return public_url
    except Exception as e:
        return f"Error generating image for '{plant_name}': {e}"


def search_herbal_knowledge_rag(query: str) -> str:
    """Searches Nicholas Culpeper's 'The Complete Herbal' ebook corpus for historical plant remedies, botanical uses, and herbal lore using Vertex AI RAG Engine.

    Args:
        query: Botanical, medicinal, or plant query to look up (e.g. 'Rosemary', 'cough remedy', 'mint', 'basil').

    Returns:
        Relevant passages retrieved from 'The Complete Herbal' RAG corpus.
    """
    import vertexai
    from vertexai.preview import rag

    try:
        vertexai.init(project=FIRESTORE_PROJECT_ID, location="us-central1")
        resp = rag.retrieval_query(
            text=query,
            rag_resources=[rag.RagResource(rag_corpus=RAG_CORPUS_NAME)],
            rag_retrieval_config=rag.RagRetrievalConfig(top_k=5),
        )
        contexts = getattr(resp.contexts, "contexts", [])
        passages = [
            c.text.strip() for c in contexts if getattr(c, "text", "").strip()
        ]
        if not passages:
            return (
                f"No relevant passages found in Culpeper's Herbal for '{query}'."
            )

        return (
            f"Passages retrieved from Culpeper's Herbal for '{query}':\n\n"
            + "\n\n---\n\n".join(passages)
        )
    except Exception as e:
        return f"Error executing RAG retrieval query: {e}"


def geocode_address(address: str) -> str:
    """Converts a street address or location name into geographic coordinates (latitude and longitude) using Google Maps Geocoding API.

    Args:
        address: Street address or city name (e.g. '1600 Amphitheatre Pkwy, Mountain View, CA' or 'Central Park, NY').

    Returns:
        Formatted address, latitude, and longitude coordinates.
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return "Error: GOOGLE_MAPS_API_KEY environment variable is not set."

    try:
        encoded_addr = urllib.parse.quote(address)
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address={encoded_addr}&key={api_key}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "GreenhouseAgent/1.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            geo_res = json.loads(resp.read().decode())

        if geo_res.get("status") != "OK" or not geo_res.get("results"):
            return f"Geocoding failed for address '{address}': {geo_res.get('status', 'ZERO_RESULTS')}"

        first = geo_res["results"][0]
        formatted = first.get("formatted_address", address)
        loc = first["geometry"]["location"]
        lat, lng = loc["lat"], loc["lng"]

        return (
            f"Geocoding Results for '{address}':\n"
            f"• Formatted Address: {formatted}\n"
            f"• Location Coordinates: Latitude {lat}, Longitude {lng}"
        )
    except Exception as e:
        return f"Error executing Geocoding request for '{address}': {e}"


def find_nearby_places(
    latitude: float,
    longitude: float,
    place_type: str = "florist",
    radius_meters: float = 5000.0,
) -> str:
    """Finds nearby places of a given type (e.g., 'florist', 'park', 'store') near coordinates using Google Places API (New).

    Args:
        latitude: Geographic latitude coordinate.
        longitude: Geographic longitude coordinate.
        place_type: Place type to search for (e.g., 'florist', 'park', 'store', 'garden_center').
        radius_meters: Search radius in meters (default: 5000.0).

    Returns:
        Formatted list of nearby places including name, address, and coordinates location.
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return "Error: GOOGLE_MAPS_API_KEY environment variable is not set."

    try:
        places_url = "https://places.googleapis.com/v1/places:searchNearby"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.primaryType",
        }
        body = json.dumps(
            {
                "includedTypes": [place_type],
                "maxResultCount": 5,
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "latitude": float(latitude),
                            "longitude": float(longitude),
                        },
                        "radius": float(radius_meters),
                    }
                },
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            places_url, data=body, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            places_res = json.loads(resp.read().decode())

        places = places_res.get("places", [])
        if not places:
            return f"No nearby places of type '{place_type}' found within {radius_meters}m of ({latitude}, {longitude})."

        results = []
        for p in places:
            name = p.get("displayName", {}).get("text", "Unknown Place")
            addr = p.get("formattedAddress", "N/A")
            loc = p.get("location", {})
            p_lat = loc.get("latitude", "N/A")
            p_lng = loc.get("longitude", "N/A")
            results.append(
                f"- Name: {name} | Address: {addr} | Location: ({p_lat}, {p_lng})"
            )

        return (
            f"Nearby Places ({place_type}) near ({latitude}, {longitude}):\n"
            + "\n".join(results)
        )
    except Exception as e:
        return f"Error executing Places (New) search: {e}"


def search_botanical_taxonomy(plant_name: str) -> str:
    """Searches official global botanical taxonomy databases (GBIF / Trefle) for scientific classification, family, genus, and scientific name.

    Args:
        plant_name: Common or scientific name of the plant (e.g. 'Monstera', 'Snake Plant', 'Fiddle Leaf Fig').

    Returns:
        Formatted official botanical taxonomy classification details.
    """
    try:
        trefle_key = os.environ.get("TREFLE_API_KEY")
        if trefle_key:
            encoded = urllib.parse.quote(plant_name)
            trefle_url = f"https://trefle.io/api/v1/plants/search?token={trefle_key}&q={encoded}"
            req = urllib.request.Request(
                trefle_url, headers={"User-Agent": "GreenhouseAgent/1.0"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                t_data = json.loads(resp.read().decode())
                if t_data.get("data"):
                    first = t_data["data"][0]
                    return (
                        f"Trefle Botanical Data for '{plant_name}':\n"
                        f"• Scientific Name: {first.get('scientific_name')}\n"
                        f"• Common Name: {first.get('common_name', 'N/A')}\n"
                        f"• Family: {first.get('family', 'N/A')}\n"
                        f"• Genus: {first.get('genus', 'N/A')}"
                    )

        encoded = urllib.parse.quote(plant_name)
        gbif_url = f"https://api.gbif.org/v1/species/search?q={encoded}&limit=1"
        req = urllib.request.Request(
            gbif_url, headers={"User-Agent": "GreenhouseAgent/1.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            results = data.get("results", [])
            if not results:
                return f"No botanical records found in GBIF registry for '{plant_name}'."

            first = results[0]
            return (
                f"Official Botanical Taxonomy for '{plant_name}' (GBIF Registry):\n"
                f"• Scientific Name: {first.get('scientificName', 'N/A')}\n"
                f"• Canonical Name: {first.get('canonicalName', 'N/A')}\n"
                f"• Family: {first.get('family', 'N/A')}\n"
                f"• Genus: {first.get('genus', 'N/A')}\n"
                f"• Kingdom: {first.get('kingdom', 'Plantae')}"
            )
    except Exception as e:
        return f"Error querying botanical taxonomy database: {e}"


def get_local_weather_and_humidity(location: str = "New York") -> str:
    """Fetches real-time temperature and relative humidity for a city using Open-Meteo API.

    Args:
        location: City name (e.g., 'New York', 'San Francisco', 'London').

    Returns:
        Current weather and humidity statement to guide plant watering.
    """
    try:
        encoded = urllib.parse.quote(location)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded}&count=1"
        req = urllib.request.Request(
            geo_url, headers={"User-Agent": "GreenhouseAgent/1.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            geo_data = json.loads(resp.read().decode())

        results = geo_data.get("results")
        if not results:
            return (
                f"Could not find coordinates for location '{location}'. "
                "Assume standard room conditions (70°F, 50% humidity)."
            )

        lat = results[0]["latitude"]
        lon = results[0]["longitude"]
        city_name = results[0].get("name", location)

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m&temperature_unit=fahrenheit"
        )
        req_w = urllib.request.Request(
            weather_url, headers={"User-Agent": "GreenhouseAgent/1.0"}
        )
        with urllib.request.urlopen(req_w, timeout=5) as resp_w:
            w_data = json.loads(resp_w.read().decode())

        current = w_data.get("current", {})
        temp = current.get("temperature_2m", 70.0)
        humidity = current.get("relative_humidity_2m", 50.0)

        return (
            f"Live Weather for {city_name}:\n"
            f"• Temperature: {temp}°F\n"
            f"• Humidity: {humidity}%\n"
            f"Advice: {'High humidity — reduce watering frequency.' if humidity > 60 else 'Low humidity — consider misting or watering more frequently.' if humidity < 40 else 'Optimal humidity level for most indoor plants.'}"
        )
    except Exception as e:
        return f"Error fetching live weather for {location}: {e}"


def upload_plant_image_to_gcs(
    plant_id: str, image_data_base64: str = "", filename: str = ""
) -> str:
    """Uploads a plant photo/image to public Cloud Storage and updates the plant in Firestore.

    Args:
        plant_id: Document ID in Firestore (e.g. 'monstera_deliciosa').
        image_data_base64: Optional base64-encoded image string. If omitted, creates a text placeholder asset.
        filename: Optional custom filename (e.g. 'monstera_photo.png').

    Returns:
        Public HTTPS URL of the uploaded image asset.
    """
    try:
        client = _get_storage_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        obj_name = (
            filename or f"{plant_id}_{int(datetime.now().timestamp())}.png"
        )
        blob = bucket.blob(obj_name)

        if image_data_base64:
            content = base64.b64decode(image_data_base64)
            blob.upload_from_string(content, content_type="image/png")
        else:
            blob.upload_from_string(
                f"Plant Image Asset for {plant_id}", content_type="text/plain"
            )

        public_url = (
            f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{obj_name}"
        )

        db = _get_firestore_client()
        doc_ref = db.collection("plants").document(plant_id)
        if doc_ref.get().exists:
            doc_ref.update({"image_url": public_url})

        return (
            f"Successfully uploaded image asset to GCS!\n"
            f"Public URL: {public_url}\n"
            f"Updated Firestore document 'plants/{plant_id}' with image_url."
        )
    except Exception as e:
        return f"Error uploading image to GCS: {e}"


def create_watering_task(
    plant_id: str, days_from_now: int = 7, notes: str = ""
) -> str:
    """Schedules an upcoming plant watering reminder in Firestore.

    Args:
        plant_id: Document ID of the plant (e.g. 'snake_plant').
        days_from_now: Number of days until next watering (e.g. 7).
        notes: Optional custom care instructions or notes.

    Returns:
        Confirmation message of the scheduled watering task.
    """
    try:
        db = _get_firestore_client()
        due_date = (datetime.now() + timedelta(days=days_from_now)).strftime(
            "%Y-%m-%d"
        )

        schedule_data = {
            "plant_id": plant_id,
            "due_date": due_date,
            "days_interval": int(days_from_now),
            "notes": notes,
            "created_at": datetime.now().isoformat(),
            "status": "pending",
        }

        task_id = f"{plant_id}_{due_date}"
        db.collection("watering_schedules").document(task_id).set(
            schedule_data
        )

        return (
            f"Successfully scheduled watering task for '{plant_id}' on {due_date}!\n"
            f"Saved to Firestore collection 'watering_schedules/{task_id}'."
        )
    except Exception as e:
        return f"Error creating watering task: {e}"


def search_plants_firestore(query: str = "") -> str:
    """Searches the Firestore 'plants' catalog for matching plants or lists all available plants.

    Args:
        query: Optional search keyword to filter plants by name or description.

    Returns:
        Formatted catalog items retrieved from Firestore.
    """
    db = _get_firestore_client()
    docs = list(db.collection("plants").stream())
    if not docs:
        return "No plants found in the Firestore inventory catalog."

    query_lower = query.lower()
    matching_plants = []
    for doc in docs:
        data = doc.to_dict()
        name = data.get("name", doc.id)
        desc = data.get("description", "")
        if (
            not query_lower
            or query_lower in name.lower()
            or query_lower in desc.lower()
        ):
            img = (
                f" | Image: {data.get('image_url')}"
                if data.get("image_url")
                else ""
            )
            matching_plants.append(
                f"- ID: {doc.id} | Name: {name} | Price: ${data.get('price_usd', 0.0):.2f} | "
                f"Light: {data.get('light_requirements', 'N/A')} | Water: {data.get('water_schedule', 'N/A')} | "
                f"Difficulty: {data.get('care_difficulty', 'N/A')}{img}"
            )

    if not matching_plants:
        return f"No plants in Firestore catalog matched query '{query}'."

    return "Firestore Plants Catalog:\n" + "\n".join(matching_plants)


def get_plant_details_firestore(plant_id: str) -> str:
    """Retrieves full details for a specific plant from Firestore by plant_id.

    Args:
        plant_id: The document ID of the plant (e.g., 'monstera_deliciosa', 'snake_plant').

    Returns:
        Formatted detailed plant information or a not found message.
    """
    db = _get_firestore_client()
    doc_ref = db.collection("plants").document(plant_id)
    doc = doc_ref.get()
    if not doc.exists:
        return f"Plant with ID '{plant_id}' was not found in Firestore."

    data = doc.to_dict()
    img_str = (
        f"\n• Image URL: {data.get('image_url')}" if data.get("image_url") else ""
    )
    return (
        f"Plant Details for '{plant_id}':\n"
        f"• Name: {data.get('name')}\n"
        f"• Price: ${data.get('price_usd', 0.0):.2f}\n"
        f"• Light: {data.get('light_requirements')}\n"
        f"• Water: {data.get('water_schedule')}\n"
        f"• Difficulty: {data.get('care_difficulty')}\n"
        f"• In Stock: {data.get('in_stock')}\n"
        f"• Description: {data.get('description')}{img_str}"
    )


def add_plant_to_firestore(
    plant_id: str,
    name: str,
    price_usd: float,
    light_requirements: str,
    water_schedule: str,
    care_difficulty: str = "Easy",
    in_stock: bool = True,
    description: str = "",
) -> str:
    """Adds a new plant or updates an existing plant in the Firestore 'plants' collection.

    Args:
        plant_id: Unique identifier for the plant document (e.g. 'peace_lily').
        name: Common name of the plant (e.g. 'Peace Lily').
        price_usd: Price in US Dollars.
        light_requirements: Sunlight needs (e.g. 'Low to bright indirect light').
        water_schedule: Watering frequency (e.g. 'Water weekly').
        care_difficulty: Difficulty level ('Beginner', 'Easy', 'Intermediate', 'Advanced').
        in_stock: Whether the plant is currently in stock.
        description: A short overview of the plant.

    Returns:
        Confirmation message of the Firestore document save.
    """
    db = _get_firestore_client()
    plant_data = {
        "plant_id": plant_id,
        "name": name,
        "price_usd": float(price_usd),
        "light_requirements": light_requirements,
        "water_schedule": water_schedule,
        "care_difficulty": care_difficulty,
        "in_stock": bool(in_stock),
        "description": description,
    }
    db.collection("plants").document(plant_id).set(plant_data)
    return f"Successfully saved plant '{name}' (ID: {plant_id}) to Firestore collection 'plants'!"


def get_plant_care(plant_name: str) -> str:
    """Provides general care instructions for a plant.

    Args:
        plant_name: Name of the plant.

    Returns:
        Care instructions string.
    """
    return get_plant_details_firestore(plant_name.lower().replace(" ", "_"))



def generate_plant_video(
    plant_name: str,
    prompt_description: str = "A short video showing the plant leaf swaying gently in a greenhouse breeze",
    tool_context: ToolContext = None,
) -> str:
    """Generates a short video for a plant item using Google's Omni model (gemini-omni-flash-preview) in the global region.
    Saves the video as an artifact using tool_context.save_artifact, uploads the video bytes directly to public Cloud Storage, and returns the public https URL.

    Args:
        plant_name: Name of the plant or item in the agent's domain (e.g. 'Monstera Deliciosa', 'Peace Lily', 'Fiddle Leaf Fig').
        prompt_description: Description of the video scene or movement to generate.
        tool_context: ToolContext automatically injected by ADK to record artifacts.

    Returns:
        The public HTTPS Cloud Storage URL (https://storage.googleapis.com/greenhouse-agent-assets-f41b114e/...) of the generated video.
    """
    try:
        client = genai.Client(
            vertexai=True,
            project=FIRESTORE_PROJECT_ID,
            location="global",
        )
        full_prompt = f"A short video of {plant_name}. {prompt_description}"
        interaction = client.interactions.create(
            model="gemini-omni-flash-preview",
            input=full_prompt,
        )

        video_b64 = getattr(getattr(interaction, "output_video", None), "data", None)
        if not video_b64:
            return f"Error: No video content generated for '{plant_name}'."

        video_bytes = base64.b64decode(video_b64)
        mime_type = "video/mp4"

        safe_name = "".join(
            c for c in plant_name.lower().replace(" ", "_") if c.isalnum() or c == "_"
        )
        filename = f"{safe_name}_video_{int(datetime.now().timestamp())}.mp4"

        # 1. Save with tool_context.save_artifact for Playground Artifacts panel
        if tool_context and hasattr(tool_context, "save_artifact"):
            part = types.Part.from_bytes(data=video_bytes, mime_type=mime_type)
            tool_context.save_artifact(filename=filename, artifact=part)

        # 2. Upload same bytes directly in memory to public GCS bucket
        storage_client = _get_storage_client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(filename)
        blob.upload_from_string(video_bytes, content_type=mime_type)

        public_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{filename}"
        return public_url
    except Exception as e:
        return f"Error generating video for '{plant_name}': {e}"


# Configure A2UI Schema Manager (v0.8) and Basic Catalog
schema_manager = A2uiSchemaManager(
    version="0.8",
    catalogs=[BasicCatalog.get_config("0.8")],
)

a2ui_instruction = schema_manager.generate_system_prompt(
    role_description=(
        "You are the Greenhouse Assistant, an expert AI plant-care advisor. "
        "You can run Python code safely in an Agent Platform sandbox environment to solve calculations, data analysis, or math tasks. "
        "You also have access to image generation with gemini-3.1-flash-lite-image (`generate_plant_image`), "
        "video generation with gemini-omni-flash-preview in the global region (`generate_plant_video`), "
        "Nicholas Culpeper's 'The Complete Herbal' ebook RAG corpus (`search_herbal_knowledge_rag`), "
        "Google Maps Geocoding (`geocode_address`), "
        "Google Places New API (`find_nearby_places`), "
        "official botanical taxonomy registry tools (`search_botanical_taxonomy`), "
        "real live weather forecasts (`get_local_weather_and_humidity`), "
        "Cloud Storage image asset uploads (`upload_plant_image_to_gcs`), "
        "watering schedule management (`create_watering_task`), "
        "and a Cloud Firestore plant catalog (`search_plants_firestore`, `get_plant_details_firestore`, `add_plant_to_firestore`). "
        "You also remember the user's personal preferences and past sessions using Vertex AI Memory Bank. "
        "When asked to generate or create an image of a plant, use `generate_plant_image`. "
        "When asked to generate a video of a plant, use `generate_plant_video`."
    ),
    workflow_description="Analyze the request, call relevant tools if needed, and return structured UI when appropriate.",
    ui_description=(
        "Keep every surface tiny and flat: ONE Card > ONE Column > a few Text rows. "
        "Never nest a Card inside a Card. "
        "Use ONLY these components: Card, Column, Row, Text, and Image. Do not use "
        "Table or Heading (unsupported), or Buttons, actions, or forms (they do "
        "nothing in adk web). "
        "You may include one Image component, but only when you have a public https "
        "URL for the image (for example the URL an image tool returns after uploading "
        "to a public bucket). Set the Image url to that exact https link, for example "
        '{"Image": {"url": {"literalString": "https://..."}}}. Never point an '
        "Image at a bare filename, an artifact name, or a non-http(s) path. If you do "
        "not have a public URL, add a short Text line noting the image instead. "
        "No markdown in text; use the usageHint property ('h1', 'h2', 'body') for "
        "headings and emphasis. "
        "Output ONLY the raw A2UI JSON array — no prose, and never wrap it in "
        "<a2a_datapart_json> tags or 'kind'/'data'/'metadata' objects."
    ),
    include_schema=True,
    include_examples=True,
)

# Configure Agent Platform sandbox code execution
sandbox_code_executor = AgentEngineSandboxCodeExecutor(
    agent_engine_resource_name=_get_agent_engine_resource_name()
)

root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    code_executor=sandbox_code_executor,
    instruction=a2ui_instruction,
    tools=[
        PreloadMemoryTool(),
        generate_plant_image,
        generate_plant_video,
        search_herbal_knowledge_rag,
        geocode_address,
        find_nearby_places,
        search_botanical_taxonomy,
        get_local_weather_and_humidity,
        upload_plant_image_to_gcs,
        create_watering_task,
        search_plants_firestore,
        get_plant_details_firestore,
        add_plant_to_firestore,
        get_plant_care,
    ],
    after_agent_callback=generate_memories_callback,
    after_model_callback=a2ui_callback,
)

app = App(
    root_agent=root_agent,
    name="app",
)

