/* ----------------------------------------------------------------------
    Manage the toggle button for switching between light/dark themes 
    ---------------------------------------------------------------------- */

(function() {
    // Attach the toggle logic to the window "hnnToggleTheme" so we can 
    // pass it to the "on click" HTML attribute when instantiating the
    // contents of title-bar (which fills the "header" gridbox in AppLayout) 
    window.hnnToggleTheme = function() {
        const c = document.querySelector('.jupyter-widgets-view') || document.body;
        if (c) {
            c.classList.toggle('dark-mode');
            const isD = c.classList.contains('dark-mode');
            const s = document.getElementById('sun-svg');
            const m = document.getElementById('moon-svg');
            
            if (s && m) {
                s.style.display = isD ? 'none' : 'block';
                m.style.display = isD ? 'block' : 'none';
            }
        }
    };
})();


/* ----------------------------------------------------------------------
    Add the "caret" buttons for scrolling in the visualization-window tabbar
    and in the figures tabbar in parameters-window > visualization-tab  
    ---------------------------------------------------------------------- */

(function() {

    const setup = (selector) => {
        const parent = document.querySelector(selector);
        if (!parent) return;

        const bar = parent.querySelector(".lm-TabBar-content");
        if (!bar || parent.querySelector(".caret-left")) return;

        const lCaret = document.createElement("div");
        lCaret.className = "tab-caret caret-left";
        lCaret.innerHTML = "&#10094;";
        const rCaret = document.createElement("div");
        rCaret.className = "tab-caret caret-right";
        rCaret.innerHTML = "&#10095;";

        const leftWall = document.createElement("div");
        const rightWall = document.createElement("div");

        let wallStyle = "position: absolute; top: 2px; ";
        wallStyle += "bottom: 0; width: 1px; ";
        wallStyle += "background-color: var(--tab-border); ";
        wallStyle += "z-index: 20; ";
        wallStyle += "pointer-events: none; display: none;";

        leftWall.style.cssText = wallStyle + "left: 0;";
        rightWall.style.cssText = wallStyle + "right: 0;";

        parent.appendChild(lCaret);
        parent.appendChild(rCaret);
        parent.appendChild(leftWall);
        parent.appendChild(rightWall);

        const update = () => {
            const canLeft = bar.scrollLeft > 1;
            const isAtEnd = bar.scrollLeft +
                bar.clientWidth >= (bar.scrollWidth - 1);

            lCaret.classList.toggle("is-visible", canLeft);
            rCaret.classList.toggle("is-visible", !isAtEnd);

            leftWall.style.display = canLeft ? "block" : "none";
            rightWall.style.display = !isAtEnd ? "block" : "none";
        };

        const doScroll = (e, amt) => {
            e.stopImmediatePropagation();
            e.stopPropagation();
            e.preventDefault();
            bar.scrollBy({ left: amt, behavior: "smooth" });
            return false;
        };

        const events = ["pointerdown", "mousedown", "click"];
        events.forEach(evtName => {
            lCaret.addEventListener(evtName, (e) =>
                doScroll(e, -150), true);
            rCaret.addEventListener(evtName, (e) =>
                doScroll(e, 150), true);
        });

        bar.addEventListener("scroll", update);
        const obs = new MutationObserver(update);
        obs.observe(bar, { childList: true, subtree: true });
        setTimeout(update, 100);
    };

    const poller = setInterval(() => {
        setup(".visualization-window .lm-TabBar");
        setup(".visualization-tab .lm-TabBar:first-of-type");
    }, 500);
})();

/* ----------------------------------------------------------------------
   disable dropdown menu displaying when no actual items are present

   note: i've *only* noticed this on Firefox, but it creates an empty
         oval on screen that looks like an erroneous box... this requires
         js as you can't target those popup boxes with CSS, unfortunately.
         We may deprecate this function later in the event that we figure
         out why this occurs under certain browser-OS combinations
   ---------------------------------------------------------------------- */

(function() {
    const blockEmpty = (e) => {
        const t = e.target;
        const isDrop = t.closest(".widget-dropdown");
        if (t.tagName === "SELECT" && isDrop) {
            if (t.childElementCount === 0) {
                e.preventDefault();
                t.focus();
            }
        }
    };

    document.addEventListener("mousedown", blockEmpty, true);

    const obs = new MutationObserver(() => {
        document.removeEventListener("mousedown", blockEmpty, true);
        document.addEventListener("mousedown", blockEmpty, true);
    });

    obs.observe(document.body, {childList: true, subtree: true});
})();