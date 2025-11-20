import importlib
import importlib.util
import re
from pathlib import Path

import streamlit as st

# Runner page: sidebar navigation that loads page modules dynamically.
st.set_page_config(page_title="IND320 App", layout="wide")

st.title("IND320 Streamlit App")
st.write("Use the sidebar to navigate between pages.")


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


# Discover pages
pages = discover_pages()

# Sidebar navigation (use a radio so all pages are visible)
st.sidebar.title("Navigation")

# Build navigation options: label -> value (value is path or 'Home')
nav_options = [('Home', 'Home')] + [(display, path) for (_m, display, path) in pages]

if not pages:
	st.sidebar.info("No pages found in `Pages/` — check that .py files exist there.")

# labels shown to the user
labels = [lbl for lbl, _val in nav_options]

# initialize page in session state
if 'page' not in st.session_state:
	st.session_state['page'] = 'Home'

# radio always shows all options; map selection back to path
choice = st.sidebar.radio('Go to', labels, index=labels.index(st.session_state.get('page', 'Home')) if st.session_state.get('page') in labels else 0)
# find corresponding value
selected = next((val for lbl, val in nav_options if lbl == choice), 'Home')
st.session_state['page'] = selected


# --- Secrets / credentials loader (Streamlit secrets preferred) ---
def load_secrets():
	"""Return a dict with keys 'mongo_user', 'mongo_pwd', and optional cassandra settings.
	Priority: st.secrets -> environment variables. Raises RuntimeError if required secrets missing.
	"""
	secrets = {}
	# Try Streamlit secrets
	try:
		s = st.secrets
	except Exception:
		s = {}

	# Mongo
	mongo_user = None
	mongo_pwd = None
	if isinstance(s, dict) and 'mongo' in s:
		m = s.get('mongo', {})
		mongo_user = m.get('user')
		mongo_pwd = m.get('password')

	# Fallback to environment variables
	import os
	if not mongo_user:
		mongo_user = os.getenv('MONGO_USER')
	if not mongo_pwd:
		mongo_pwd = os.getenv('MONGO_PWD')

	if not (mongo_user and mongo_pwd):
		# Do not raise immediately here; return Nones so caller can decide how to proceed
		return {'mongo_user': None, 'mongo_pwd': None}

	# Cassandra optional settings
	cass_host = None
	cass_port = None
	if isinstance(s, dict) and 'cassandra' in s:
		c = s.get('cassandra', {})
		cass_host = c.get('host')
		cass_port = c.get('port')
	if not cass_host:
		cass_host = os.getenv('CASSANDRA_HOST', '127.0.0.1')
	if not cass_port:
		cass_port = int(os.getenv('CASSANDRA_PORT', '9042'))

	return {'mongo_user': mongo_user, 'mongo_pwd': mongo_pwd, 'cass_host': cass_host, 'cass_port': cass_port}


# show secret presence in sidebar (masked)
_secrets = load_secrets()
if _secrets.get('mongo_user'):
	st.sidebar.write('Mongo credentials: :white_check_mark:')
else:
	st.sidebar.warning('Mongo credentials: missing (set .streamlit/secrets or env vars MONGO_USER / MONGO_PWD)')



def load_module_from_path(path_str: str, module_alias: str):
	"""Import a module given a file path using importlib and return the module."""
	spec = importlib.util.spec_from_file_location(module_alias, path_str)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


current = st.session_state.get('page', 'Home')

if current == "Home":
	# Show a simple home page with a preview (load Data_loader if available)
	st.header("Home")
	try:
		from StreamlitApplication.Data_loader import load_data

		df = load_data()
		st.subheader("Preview (first 10 rows)")
		st.dataframe(df.head(10), use_container_width=True)
		st.write(f"Data has {len(df)} rows and {len(df.columns)} columns.")
	except Exception as e:
		st.warning("Could not load data preview: " + str(e))

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
		st.header(display)
		try:
			module = load_module_from_path(path, mod_name)
			# Call main() if present, otherwise importing executed the page already
			if hasattr(module, "main") and callable(module.main):
				module.main()
		except Exception as e:
			st.error(f"Error loading page {display}: {e}")