(function () {
  function formatStars(count) {
    if (typeof count !== "number") return "";
    if (count < 1000) return String(count);
    return (count / 1000).toFixed(count < 10000 ? 1 : 0).replace(/\.0$/, "") + "k";
  }

  var el = document.querySelector("[data-stars-url]");
  if (!el) return;

  fetch(el.getAttribute("data-stars-url"))
    .then(function (response) {
      if (!response.ok) throw new Error("GitHub API returned " + response.status);
      return response.json();
    })
    .then(function (data) {
      var text = formatStars(data.stargazers_count);
      if (text) el.textContent = text;
    })
    .catch(function () {
      el.textContent = "";
    });
})();
