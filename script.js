const showPassword = document.getElementById("showPassword");
const password = document.getElementById("password");

showPassword.addEventListener("change", function () {
    password.type = this.checked ? "text" : "password";
});