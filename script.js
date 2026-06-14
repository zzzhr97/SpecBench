const copyButtons = document.querySelectorAll("[data-copy-target]");

copyButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    const targetId = button.getAttribute("data-copy-target");
    const target = targetId ? document.getElementById(targetId) : null;

    if (!target) {
      return;
    }

    try {
      await navigator.clipboard.writeText(target.innerText);
      const originalLabel = button.textContent;
      button.textContent = "Copied";
      window.setTimeout(() => {
        button.textContent = originalLabel;
      }, 1600);
    } catch {
      button.textContent = "Select text";
    }
  });
});
