/** Run before ThemeToggle loads so the first paint respects stored preference. */
if (!localStorage.getItem("theme")) {
  localStorage.setItem("theme", "dark");
}
