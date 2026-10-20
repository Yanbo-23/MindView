(() => {
  const links = document.querySelectorAll('.contents a');
  const sections = [...links].map((link) => document.querySelector(link.getAttribute('href'))).filter(Boolean);
  if (!('IntersectionObserver' in window)) return;
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      const link = document.querySelector(`.contents a[href="#${entry.target.id}"]`);
      if (link && entry.isIntersecting) {
        links.forEach((item) => item.classList.remove('current'));
        link.classList.add('current');
      }
    });
  }, { rootMargin: '-20% 0px -70% 0px' });
  sections.forEach((section) => observer.observe(section));
})();
