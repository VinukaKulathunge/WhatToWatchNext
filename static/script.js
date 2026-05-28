document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('search-form');
    const searchInput = document.getElementById('search-input');
    const autocompleteList = document.getElementById('autocomplete-list');
    const spinner = document.getElementById('loading-spinner');
    const errorMsg = document.getElementById('error-message');
    const resultsContainer = document.getElementById('results-container');
    const seedMovieInfo = document.getElementById('seed-movie-info');
    const moviesGrid = document.getElementById('movies-grid');

    let currentSelectedTmdbId = null;

    // Debounce function to avoid hitting the API too often while typing
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Handle Autocomplete Search
    const fetchAutocomplete = async (query) => {
        if (!query) {
            autocompleteList.classList.add('hidden');
            return;
        }
        try {
            const res = await fetch(`/search?q=${encodeURIComponent(query)}`);
            if (res.ok) {
                const movies = await res.json();
                renderAutocomplete(movies);
            }
        } catch (e) {
            console.error("Autocomplete failed", e);
        }
    };

    const renderAutocomplete = (movies) => {
        autocompleteList.innerHTML = '';
        if (movies.length === 0) {
            autocompleteList.classList.add('hidden');
            return;
        }

        movies.forEach(movie => {
            const item = document.createElement('div');
            item.className = 'autocomplete-item';
            
            // Basic HTML layout for autocomplete item
            item.innerHTML = `
                <div class="movie-poster-placeholder" style="width: 40px; height: 60px; font-size: 1rem; border-radius: 4px;" id="auto-poster-${movie.tmdb_id}">
                    <i class="fas fa-film"></i>
                </div>
                <span>${movie.title}</span>
            `;

            // Click event for the autocomplete item
            item.addEventListener('click', () => {
                searchInput.value = movie.title;
                currentSelectedTmdbId = movie.tmdb_id; // Store ID for exact match
                autocompleteList.classList.add('hidden');
                
                // Automatically submit the form
                searchForm.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
            });

            autocompleteList.appendChild(item);

            // Fetch poster asynchronously for the autocomplete dropdown
            fetchPoster(movie.tmdb_id).then(posterUrl => {
                if (posterUrl) {
                    const placeholder = document.getElementById(`auto-poster-${movie.tmdb_id}`);
                    if(placeholder) {
                        placeholder.innerHTML = `<img src="${posterUrl}" style="width: 40px; height: 60px; object-fit: cover; border-radius: 4px;">`;
                    }
                }
            });
        });

        autocompleteList.classList.remove('hidden');
    };

    // Attach input event with debounce
    searchInput.addEventListener('input', debounce((e) => {
        currentSelectedTmdbId = null; // Clear exact match if user types
        fetchAutocomplete(e.target.value.trim());
    }, 300));

    // Hide autocomplete when clicking outside
    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !autocompleteList.contains(e.target)) {
            autocompleteList.classList.add('hidden');
        }
    });

    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = searchInput.value.trim();
        
        if (!query) return;

        if (!currentSelectedTmdbId) {
            errorMsg.textContent = "Please select a movie from the dropdown list to ensure accurate recommendations.";
            errorMsg.classList.remove('hidden');
            return;
        }

        // Reset state
        errorMsg.classList.add('hidden');
        resultsContainer.classList.add('hidden');
        autocompleteList.classList.add('hidden');
        spinner.classList.remove('hidden');
        
        try {
            // Build request payload
            const payload = {
                tmdb_id: currentSelectedTmdbId
            };

            const response = await fetch('/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail?.detail || data.detail?.error || 'Failed to fetch recommendations. Try another movie.');
            }

            displayResults(data);
        } catch (err) {
            errorMsg.textContent = err.message;
            errorMsg.classList.remove('hidden');
        } finally {
            spinner.classList.add('hidden');
        }
    });

    async function fetchPoster(tmdbId) {
        // Fetch real poster path from our new TMDB proxy endpoint
        try {
            const res = await fetch(`/poster/${tmdbId}`);
            if (res.ok) {
                const data = await res.json();
                return data.poster_url;
            }
        } catch(e) {
            console.error("Could not fetch poster", e);
        }
        return null;
    }

    async function displayResults(data) {
        // Set seed movie info layout
        seedMovieInfo.innerHTML = `
            <h2>Recommendations based on</h2>
            <div style="display: flex; flex-direction: column; align-items: center; gap: 1rem; margin-top: 1rem;">
                <div id="seed-poster" class="movie-poster-placeholder" style="width: 120px; height: 180px; font-size: 2rem; border-radius: 8px;">
                    <i class="fas fa-film"></i>
                </div>
                <p style="font-size: 1.8rem; font-weight: 700; color: white; margin: 0;">"${data.seed_title}"</p>
            </div>
        `;

        // Fetch poster for the seed movie
        fetchPoster(data.seed_tmdb_id).then(posterUrl => {
            if (posterUrl) {
                const seedPoster = document.getElementById('seed-poster');
                if (seedPoster) {
                    seedPoster.innerHTML = `<img src="${posterUrl}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 8px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5);">`;
                }
            }
        });

        // Clear previous results
        moviesGrid.innerHTML = '';

        // Render cards
        for (const movie of data.results) {
            const card = document.createElement('div');
            card.className = 'movie-card';

            const tmdbUrl = `https://www.themoviedb.org/movie/${movie.tmdb_id}`;

            let graphExplanationHtml = '';
            let graphBadgeHtml = '';

            if (movie.graph_boost > 0 && movie.graph_explanation?.has_connection) {
                graphBadgeHtml = `
                    <div class="score-badge graph-badge" title="Knowledge Graph Boost">
                        <i class="fas fa-project-diagram"></i> +${movie.graph_boost.toFixed(2)} Graph
                    </div>
                `;
                graphExplanationHtml = `
                    <div class="graph-explanation">
                        <i class="fas fa-link"></i> ${movie.graph_explanation.connection_detail}
                    </div>
                `;
            }

            let expText = movie.explanation;
            expText = expText.replace(/Recommended because the synopsis is/gi, 'Synopsis is');
            
            card.innerHTML = `
                <div class="movie-poster-placeholder" id="poster-${movie.tmdb_id}">
                    <i class="fas fa-film"></i>
                </div>
                <div class="movie-content">
                    <a href="${tmdbUrl}" target="_blank" style="text-decoration: none; color: inherit;">
                        <h3 class="movie-title">${movie.title} <i class="fas fa-external-link-alt" style="font-size: 0.8rem; color: #64748b; margin-left: 0.5rem;"></i></h3>
                    </a>
                    <div class="movie-scores">
                        <div class="score-badge" title="Semantic Similarity">
                            <i class="fas fa-brain"></i> ${(movie.cosine_similarity * 100).toFixed(0)}% Match
                        </div>
                        ${graphBadgeHtml}
                        <div class="score-badge" style="background: rgba(236, 72, 153, 0.1); color: #f472b6; border-color: rgba(236, 72, 153, 0.2);" title="Final Score">
                            <i class="fas fa-star"></i> Score ${movie.final_score.toFixed(2)}
                        </div>
                    </div>
                    <p class="movie-explanation">${expText}</p>
                    ${graphExplanationHtml}
                </div>
            `;
            moviesGrid.appendChild(card);

            // Fetch poster asynchronously
            fetchPoster(movie.tmdb_id).then(posterUrl => {
                if (posterUrl) {
                    const placeholder = document.getElementById(`poster-${movie.tmdb_id}`);
                    if (placeholder) {
                        placeholder.innerHTML = `<img src="${posterUrl}" style="width: 100%; height: 100%; object-fit: cover; position: absolute; top: 0; left: 0;">`;
                    }
                }
            });
        }

        resultsContainer.classList.remove('hidden');
    }
});
