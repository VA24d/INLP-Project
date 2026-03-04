document.addEventListener("DOMContentLoaded", () => {

    /* Smooth Scrolling for anchor links */
    document.querySelectorAll('.nav-links a, .hero-buttons a').forEach(anchor => {
        anchor.addEventListener('click', function (e) {

            // Allow external links to pass through
            if (this.getAttribute('href').startsWith('http')) return;

            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);

            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 70, // Offset for fixed navbar
                    behavior: 'smooth'
                });
            }
        });
    });

    /* Ambient Glow orb movement effect on Hero */
    const hero = document.querySelector('.hero');
    const orb = document.querySelector('.glow-orb');

    if (hero && orb) {
        hero.addEventListener('mousemove', (e) => {
            const x = e.clientX;
            const y = e.clientY - hero.getBoundingClientRect().top;

            // Smoothing movement
            orb.animate({
                left: `${x}px`,
                top: `${y}px`
            }, { duration: 3000, fill: "forwards" });
        });
    }

    /* Subtle reveal animations on scroll */
    const observerOptions = {
        threshold: 0.1,
        rootMargin: "0px 0px -50px 0px"
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = 1;
                entry.target.style.transform = "translateY(0)";
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Apply starting styles and observe
    document.querySelectorAll('.card, .finding-box, .code-comparison, .section-title, .section-desc').forEach(el => {
        el.style.opacity = 0;
        el.style.transform = "translateY(20px)";
        el.style.transition = "opacity 0.6s ease-out, transform 0.6s ease-out";
        observer.observe(el);
    });

});
