(() => {
  const reveals = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries, instance) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          instance.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });
    reveals.forEach((element) => observer.observe(element));
  } else {
    reveals.forEach((element) => element.classList.add('is-visible'));
  }

  const filters = document.querySelectorAll('.filter-button');
  const cards = document.querySelectorAll('.task-card');
  filters.forEach((button) => {
    button.addEventListener('click', () => {
      const selected = button.dataset.filter;
      filters.forEach((item) => item.classList.toggle('is-active', item === button));
      cards.forEach((card) => {
        const matches = selected === 'all' || card.dataset.group === selected || (selected === 'geometry' && card.classList.contains('semantics') && card.querySelector('.task-tag')?.textContent.includes('GEOMETRY'));
        card.classList.toggle('is-hidden', !matches);
      });
    });
  });
})();
