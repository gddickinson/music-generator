// Entry point: registers the lazily loaded pages, sets the footer mode text, draws the background and starts the router.
import { start, route, lazy, app, isStatic, $ } from './core.js';
import { drawBackground } from './background.js';

const PAGES = { 'home.js': ['home'], 'compose.js': ['compose'], 'fractal.js': ['fractal'], 'verify.js': ['verify'],
  'about.js': ['about'] };
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));

route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic() ? 'static site: the composer runs in your browser'
  : 'local app (Python server): the composer runs in your browser, Verify can also run the Python';
drawBackground($('#bg'));
start();
