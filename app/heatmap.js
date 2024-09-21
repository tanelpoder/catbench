// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2024 Tanel Poder
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

function getQueryParams() {
    const params = {};
    window.location.search.substr(1).split('&').forEach(function (item) {
        const [key, value] = item.split('=');
        if (key) {
            params[key] = decodeURIComponent(value);
        }
    });
    return params;
}

const params = getQueryParams();
const dataset = params.dataset || 'cats';  // Default to 'cats' if not specified

document.getElementById('datasetName').textContent = 'Dataset: ' + dataset.charAt(0).toUpperCase() + dataset.slice(1);

let dataCache = {};
let hideTooltipTimeout;

// Update heatmap based on checkbox states
function updateHeatmap() {
    const isSorted = document.getElementById('sortedCheckbox').checked;
    const isNormalized = document.getElementById('normalizedCheckbox').checked;

    let url = '/data?dataset=' + encodeURIComponent(dataset) + '&';
    if (isSorted) url += 'sorted=1&';
    if (isNormalized) url += 'normalized=1&';

    const cacheKey = `dataset=${dataset}&sorted=${isSorted}&normalized=${isNormalized}`;

    if (dataCache[cacheKey]) {
        drawHeatmap(dataCache[cacheKey]);
    } else {
        fetch(url)
            .then(response => response.json())
            .then(data => {
                dataCache[cacheKey] = data; // Update the cache
                drawHeatmap(data);
            });
    }
}

// Function to draw the heatmap
function drawHeatmap(data) {
    const numRows = data.length;
    const numColumns = data[0].embedding.length;

    // Set up canvas
    const canvas = document.getElementById('heatmapCanvas');
    canvas.width = numColumns;
    canvas.height = numRows;
    const ctx = canvas.getContext('2d');

    // Create color scale
    const colorScale = d3.scaleSequential(d3.interpolateRdBu)
        .domain([1, 0]);

    // Render heatmap
    const imageData = ctx.createImageData(numColumns, numRows);
    let index = 0;
    for (let row = 0; row < numRows; row++) {
        for (let col = 0; col < numColumns; col++) {
            const value = data[row].embedding[col];
            const color = d3.color(colorScale(value));

            imageData.data[index++] = color.r; // Red
            imageData.data[index++] = color.g; // Green
            imageData.data[index++] = color.b; // Blue
            imageData.data[index++] = 255;     // Alpha
        }
    }
    ctx.putImageData(imageData, 0, 0);

    // Tooltip setup
    const tooltip = document.getElementById('tooltip');
    canvas.onmousemove = function(event) {
        clearTimeout(hideTooltipTimeout);

        const rect = canvas.getBoundingClientRect();
        const x = Math.floor((event.clientX - rect.left) * (canvas.width / rect.width));
        const y = Math.floor((event.clientY - rect.top) * (canvas.height / rect.height));

        if (x >= 0 && x < numColumns && y >= 0 && y < numRows) {
            const filename = data[y].filename;
            const value = data[y].embedding[x];

            // Show tooltip with embedding info and image
            tooltip.style.display = 'block';
            tooltip.style.left = `${event.pageX + 10}px`;
            tooltip.style.top = `${event.pageY + 10}px`;
            tooltip.innerHTML = `
                <strong>Row:</strong> ${y}, <strong>Col:</strong> ${x}<br>
                <strong>Value:</strong> ${value.toFixed(3)}<br>
                <strong>Filename:</strong> ${filename}<br>
                <img src="/image/${encodeURIComponent(dataset)}/${encodeURIComponent(filename)}" alt="Image" onerror="this.style.display='none';">
            `;
        } else {
            hideTooltipTimeout = setTimeout(() => {
                tooltip.style.display = 'none';
            }, 100); // Delay in milliseconds
        }
    };

    canvas.onmouseleave = function() {
        hideTooltipTimeout = setTimeout(() => {
            tooltip.style.display = 'none';
        }, 100); // Delay in milliseconds
    };
}

// Add event listeners for checkboxes to update the heatmap when toggled
document.getElementById('sortedCheckbox').addEventListener('change', updateHeatmap);
document.getElementById('normalizedCheckbox').addEventListener('change', updateHeatmap);

// Initial load
updateHeatmap();
