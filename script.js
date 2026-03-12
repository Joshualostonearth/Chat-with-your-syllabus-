// Tab switching

const tabs = document.querySelectorAll(".tab");
const contents = document.querySelectorAll(".tab-content");

tabs.forEach(tab => {
tab.addEventListener("click", () => {

tabs.forEach(t => t.classList.remove("active"));
contents.forEach(c => c.classList.remove("active"));

tab.classList.add("active");

document
.getElementById(tab.dataset.tab)
.classList.add("active");

});
});


// Chat functionality

function sendMessage(){

const input = document.getElementById("userInput");
const chatBox = document.getElementById("chatBox");

if(input.value.trim()==="") return;

const userMessage = document.createElement("div");
userMessage.className="chat-message user";
userMessage.innerText=input.value;

chatBox.appendChild(userMessage);


// fake AI response

const botMessage = document.createElement("div");
botMessage.className="chat-message bot";
botMessage.innerText="Searching syllabus...";

setTimeout(()=>{

botMessage.innerText=
"Answer: Gradient descent minimizes the loss function.\nSource: Page 42";

},1000);

chatBox.appendChild(botMessage);

input.value="";

chatBox.scrollTop=chatBox.scrollHeight;

}


// Upload simulation

document.getElementById("pdfUpload").addEventListener("change", function(){

document.getElementById("statusText").innerText="Processing document...";

setTimeout(()=>{

document.getElementById("statusText").innerText="Ready";

document.getElementById("pages").innerText="72";
document.getElementById("chunks").innerText="184";
document.getElementById("topics").innerText="12";

},2000);

});