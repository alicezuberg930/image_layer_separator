() => {
  window.__layerDividerPencils?.observer.disconnect();

  let pending = false;
  let requestNumber = 0;
  const requestAction = (action, index, filename) => {
    const input = document.querySelector('#layer-edit-request textarea, #layer-edit-request input');
    if (!input) return;
    input.value = JSON.stringify({ action, index, filename, request: `${Date.now()}-${++requestNumber}` });
    input.dispatchEvent(new Event('input', { bubbles: true }));
  };

  const actionButton = (parent, action, index, filename, preview = false) => {
    const className = action === 'edit' ? 'layer-edit-pencil' : 'layer-delete-button';
    let button = parent.querySelector(`:scope > .${className}`);
    if (!button) {
      button = document.createElement('button');
      button.type = 'button';
      button.className = `layer-action-button ${className}` + (preview ? ' layer-action-button--preview' : '');
      if (action === 'edit') {
        button.textContent = '\u270e';
      } else {
        button.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/></svg>';
      }
      button.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        requestAction(action, Number(button.dataset.layerIndex), button.dataset.layerFilename);
      });
      button.addEventListener('keydown', (event) => event.stopPropagation());
      parent.appendChild(button);
    }
    if (button.dataset.layerIndex !== String(index)) button.dataset.layerIndex = String(index);
    if (button.dataset.layerFilename !== filename) button.dataset.layerFilename = filename || '';
    button.disabled = !filename;
    const name = `${action === 'edit' ? 'Edit' : 'Delete'} layer: ${filename || index + 1}`;
    if (button.getAttribute('aria-label') !== name) {
      button.setAttribute('aria-label', name);
      button.title = name;
    }
  };

  const controls = (parent, index, filename, preview = false) => {
    actionButton(parent, 'edit', index, filename, preview);
    actionButton(parent, 'delete', index, filename, preview);
  };

  const decorate = () => {
    pending = false;
    const gallery = document.getElementById('layer-gallery');
    if (!gallery) return;

    // Gradio owns the thumbnail buttons. Wrap them so our sibling action buttons
    // are valid interactive controls without replacing Gradio's event handlers.
    gallery.querySelectorAll('.layer-edit-tile').forEach((tile) => {
      if (!tile.querySelector('.thumbnail-lg')) tile.remove();
    });
    const cards = Array.from(gallery.querySelectorAll('.grid-container .thumbnail-lg'));
    cards.forEach((card, index) => {
      let tile = card.parentElement;
      if (!tile.classList.contains('layer-edit-tile')) {
        tile = document.createElement('div');
        tile.className = 'layer-edit-tile';
        card.parentElement.insertBefore(tile, card);
        tile.appendChild(card);
      }
      controls(tile, index, card.querySelector('img')?.alt);
    });

    const toolbar = gallery.querySelector('.preview .icon-buttons');
    const selected = cards.findIndex((card) => card.classList.contains('selected'));
    if (toolbar && selected >= 0) {
      controls(toolbar, selected, cards[selected].querySelector('img')?.alt, true);
    } else if (toolbar) {
      toolbar.querySelectorAll('.layer-action-button').forEach((button) => button.remove());
    }
  };

  const refresh = () => {
    if (!pending) {
      pending = true;
      requestAnimationFrame(decorate);
    }
  };
  const observer = new MutationObserver(refresh);
  observer.observe(document.getElementById('layer-preview-group') || document.body, {
    childList: true, subtree: true, characterData: true,
    attributes: true, attributeFilter: ['class', 'alt', 'title'],
  });
  window.__layerDividerPencils = { refresh, observer };
  refresh();
  return [];
}
