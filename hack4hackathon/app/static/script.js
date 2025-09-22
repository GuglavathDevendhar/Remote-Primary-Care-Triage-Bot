document.getElementById("prediction-form").addEventListener("submit", function(event){
    let inputs = document.querySelectorAll("input");
    let valid = true;

    inputs.forEach(input => {
        if(input.value === "") valid = false;
    });

    if(!valid){
        alert("Please fill all fields before submitting!");
        event.preventDefault();
    }
});
