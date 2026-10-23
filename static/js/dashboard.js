function csrfToken() {
  const meta = document.querySelector('meta[name="csrf-token"]');
  return meta ? meta.getAttribute("content") : "";
}

function updateOffender(category, folder, field, value) {
  fetch("/update_offender", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRF-Token": csrfToken(),
    },
    body: JSON.stringify({ category, folder, field, value }),
  })
    .then(async (response) => {
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.status !== "success") {
        window.alert("Error: " + (data.message || "Update failed"));
        return;
      }
      if (field === "fine_applied") {
        document.querySelectorAll(".image-card").forEach((card) => {
          if (card.dataset.category === category && card.dataset.folder === folder) {
            card.classList.toggle("fine-applied", value);
            card.classList.toggle("fine-pending", !value);
          }
        });
      }
    })
    .catch(() => {
      window.alert("Failed to update offender data");
    });
}

document.querySelectorAll("form").forEach((form) => {
  form.addEventListener("submit", (event) => event.preventDefault());
});

document.querySelectorAll(".editable-field").forEach((input) => {
  input.addEventListener("change", () => {
    updateOffender(input.dataset.category, input.dataset.folder, input.dataset.field, input.value);
  });
});

document.querySelectorAll(".editable-checkbox").forEach((box) => {
  box.addEventListener("change", () => {
    updateOffender(box.dataset.category, box.dataset.folder, box.dataset.field, box.checked);
  });
});
