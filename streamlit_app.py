import importlib
import importlib.util
import re
from pathlib import Path
import streamlit as st
def check_data_requirements(require_weather=False, require_coordinates=False, require_energy=False):
    """
    Sjekker om nødvendige data er tilgjengelig i session state.
    Stopper kjøringen og viser en advarsel hvis data mangler.
    
    Returnerer: True hvis alle krav er møtt, False ellers.
    """
    missing = []
    
    # Sjekk etter koordinater (satt av kartet)
    if require_coordinates:
        if ('map_lat' not in st.session_state or 
            st.session_state.get('map_lat') is None or 
            st.session_state.get('map_lat') == 63.5): # Bruk initialverdien som sjekk
            missing.append("lokasjonsvalg (klikk på kartet)")
    
    # Sjekk etter værdata (satt av side 2)
    if require_weather:
        if 'weather_data' not in st.session_state or st.session_state.weather_data is None:
            missing.append("nedlasting av værdata")
    
    # Sjekk etter energidata (satt av en data-loader)
    if require_energy:
        # Hvis du bruker separate funksjoner for produksjon/forbruk, sjekk begge:
        if ('production_data' not in st.session_state and 
            'consumption_data' not in st.session_state):
            missing.append("energidata")
    
    if missing:
        st.warning(f"⚠️ Mangler: {', '.join(missing)}. Vennligst besøk **Kartvisualisering** eller **Weather Data Downloader** siden først.")
        return False
    return True

# Runner page: sidebar navigation that loads page modules dynamically.
st.set_page_config(page_title="IND320 App", layout="wide")

# Helper: discover page modules under two possible folders
def discover_pages():
	"""Return a list of (module_name, display_name, path) for pages found."""
	candidates = []
	roots = [Path(__file__).parent / "Pages", Path(__file__).parent / "DataApp" / "Pages"]
	for root in roots:
		if not root.exists():
			continue
		for py in sorted(root.glob("*.py")):
			name = py.stem
			# create a display name: remove leading digits/underscores/hyphens then prettify
			display_raw = re.sub(r'^[\d_\-\s]+', '', name)
			display = display_raw.replace("_", " ").title()
			# module path for importlib (use a dynamic spec)
			module_name = f"{py.parent.name}.{name}"
			candidates.append((module_name, display, str(py)))
	return candidates


# Simple emoji chooser based on page display name keywords (fallback)
def emoji_for(display: str) -> str:
	mapping = {
		'data': '📊',
		'table': '📋',
		'visual': '📈',
		'chart': '📈',
		'plot': '📈',
		'map': '🗺️',
		'analysis': '🧠',
		'model': '🤖',
		'predict': '🔮',
		'home': '🏠',
		'about': 'ℹ️',
		'settings': '⚙️',
		'upload': '📤',
		'download': '📥',
		'dashboard': '📊',
		'report': '📝',
		'image': '🖼️',
		'text': '✍️',
		'audio': '🔊',
		'video': '🎬',
		'timeline': '📅'
	}
	low = display.lower()
	for key, emoji in mapping.items():
		if key in low:
			return emoji
	# fallback
	return '🔹'


# Discover pages
pages = discover_pages()

# Build a palette and assign a (different) emoji to each discovered page (by path)
emoji_palette = [
	"📈", "🎯","📊", "🔬", "🔍"
]
emoji_map = {}
for i, (_mod, display, path) in enumerate(pages):
	emoji_map[path] = emoji_palette[i % len(emoji_palette)]


# Sidebar navigation
st.sidebar.title("Navigation 🧭")

# Initialize session state for current page
if 'page' not in st.session_state:
	st.session_state['page'] = 'Home'

# Home button with emoji
if st.sidebar.button("🏠 Home", key="nav_home"):
	st.session_state['page'] = 'Home'

# One button per discovered page (each page gets a distinct emoji from the palette)
for i, (_mod, display, path) in enumerate(pages):
	emoji = emoji_map.get(path, emoji_for(display))
	if st.sidebar.button(f"{emoji} {display}", key=f"nav_{i}"):
		st.session_state['page'] = path


def load_module_from_path(path_str: str, module_alias: str):
	"""Import a module given a file path using importlib and return the module."""
	spec = importlib.util.spec_from_file_location(module_alias, path_str)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


current = st.session_state.get('page', 'Home')

if current == "Home":
	# App title only shown on the Home page
	st.title("IND320 Streamlit App 🚀")

	st.write(
		"Welcome to the IND320 Streamlit App. This application collects a set of pages for "
		"exploring data, visualizations, and interactive analyses. Use the sidebar to open any page."
	)
	st.write("What to expect on each page:")
	if pages:
		for _mod, display, path in pages:
			emoji = emoji_map.get(path, emoji_for(display))
			# Use Markdown to make the site names bold and include the chosen emoji
			st.markdown(f"- {emoji} **{display}**: Open this page to access tools, visualizations, tables, or analyses related to {display.lower()}.")
	else:
		st.write("- No additional pages were discovered. Add Python files to the Pages folder to create pages.")
else:
	# Find the page by file path stored in session state
	match = None
	for mod_name, display, path in pages:
		if path == current:
			match = (mod_name, display, path)
			break
	if match is None:
		st.error("Page not found or not discovered")
	else:
		mod_name, display, path = match
		# Show the page header with the emoji assigned from the palette
		st.header(f"{emoji_map.get(path, emoji_for(display))} {display}")
		try:
			module = load_module_from_path(path, mod_name)
			# Call main() if present, otherwise importing executed the page already
			if hasattr(module, "main") and callable(module.main):
				module.main()
		except Exception as e:
			st.error(f"Error loading page {display}: {e}")
			# Hide the page header in the main area for non-Home pages by injecting CSS.
			# This keeps the navigation in the sidebar visible while removing the big title rendered by st.header.
			if current != "Home":
				st.markdown(
					"""
					<style>
					/* Hide the first header (where st.header renders) on subpages */
					[data-testid="stApp"] h2:first-of-type { display: none !important; }
					</style>
					""",
					unsafe_allow_html=True,
				)