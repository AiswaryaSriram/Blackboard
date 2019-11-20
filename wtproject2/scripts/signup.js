signup={
	xhr:new XMLHttpRequest(),

	sendToDB:function send()
	{
		//alert("called send to db");
		//alert(this);
		
		this.xhr.open("POST","http://localhost:5000/signup",true);
		this.xhr.setRequestHeader("Content-type","application/x-www-form-urlencoded");
		this.xhr.onreadystatechange=this.success;
		name_obj=document.getElementById("name");
		email_obj=document.getElementById("email");
		password1_obj=document.getElementById("password1"); 
		encodedpwd= window.btoa(password1_obj.value);

		string_of_data="name="+name_obj.value+"&"+"email="+email_obj.value+"&password="+encodedpwd;
		//alert(string_of_data);
		this.xhr.send(string_of_data); 

	},
	success:function success()
	{
		//alert("called success but not satifying status etc");

		if(this.readyState==4 && this.status==200)
		{
			alert("SIGNED UP!!!");
			email_obj=document.getElementById("email");
			password1_obj=document.getElementById("password1"); 
			encodedpwd= window.btoa(password1_obj.value);
			localStorage.setItem("email",email_obj.value);
			localStorage.setItem("password",encodedpwd);
			localStorage.setItem("status","logged_in");
			window.location=  "our_dashboard.html"
		}
	},
	check:function verify_form_data(e)
	{
		//alert("checking ");

		e.preventDefault();
		
		name_obj=document.getElementById("name");
		email_obj=document.getElementById("email");
		password1_obj=document.getElementById("password1");
		password2_obj=document.getElementById("password2");
		var obj= /^([a-zA-Z0-9_.+-])+\@(([a-zA-Z0-9-])+\.)+([a-zA-Z0-9]{2,4})+$/.exec(email_obj.value);
		//alert(typeof(password1_obj.value));
		//alert(obj[0]);
		if (name_obj.value==""){
			alert("Name not entered");
		} 
		else if( email_obj.value=="" ){
			alert("Email not entered");
		}
		else if(obj==null || obj[0]!=email_obj.value){
			alert("Email not valid");
			
		}
		else if (password1_obj.value==""){
			alert("password not entered");
		}
		else if (password2_obj.value==""){
			alert("Please reconfirm password");
		}
		else if (password1_obj.value!=password2_obj.value ){
			alert("Passwords do not match");
		}
		
		else{
			signup.sendToDB();
	
		}
	}
};

login= {
	xhr:new XMLHttpRequest(),
	check:function verify_form_data(e)
	{
		//alert("checking ");

		e.preventDefault();
		
		
		email_obj=document.getElementById("login_email");
		password_obj=document.getElementById("login_password");
		
		var obj= /^([a-zA-Z0-9_.+-])+\@(([a-zA-Z0-9-])+\.)+([a-zA-Z0-9]{2,4})+$/.exec(email_obj.value);
		//alert(typeof(password1_obj.value));
		//alert(obj[0]);
		
		if( email_obj.value=="" ){
			alert("Email not entered");
		}
		else if(obj==null || obj[0]!=email_obj.value){
			alert("Email not valid");
			
		}
		else if (password_obj.value==""){
			alert("password not entered");
		}
	
	
		else{
			login.sendToDB();
	
		}
	},
	sendToDB:function send()
	{
		//alert("called send to db");
		//alert(this);
		
		this.xhr.open("POST","http://localhost:5000/login",true);
		this.xhr.setRequestHeader("Content-type","application/x-www-form-urlencoded");
		this.xhr.onreadystatechange=this.success;
		email_obj=document.getElementById("login_email");
		password_obj=document.getElementById("login_password"); 
		encodedpwd= window.btoa(password_obj.value);

		string_of_data="email="+email_obj.value+"&password="+encodedpwd;
		//alert(string_of_data);
		this.xhr.send(string_of_data); 

	},

	success:function success()
	{
		//alert("called success but not satifying status etc");

		if(this.readyState==4 && this.status==200)
		{
			if (this.responseText=="user does not exist"){
				alert("Credentials are wrong");
			}
			else{
				alert("LOGGED IN!!!");
				login_email_obj=document.getElementById("login_email");
				login_password_obj=document.getElementById("login_password"); 
				encodedpwd= window.btoa(login_password_obj.value);
				localStorage.removeItem("email");
				localStorage.removeItem("password");
				localStorage.removeItem("status");
				localStorage.setItem("email",login_email_obj.value);
				localStorage.setItem("password",encodedpwd);
				localStorage.setItem("status","logged_in");
				window.location= "our_dashboard.html"
			}
		}

	}

}
function goto_signup1(){

	document.getElementById("signup_form").style.display="";
	document.getElementById("login_form").style.display="none";
}

//Event listeners for sign up buton
signup_btn=document.getElementById("signup");
signup_btn.addEventListener("click",signup.check,false);

document.getElementById("signup_form").style.display="none";
goto_signup=document.getElementById("goto_signup");
goto_signup.addEventListener("click", goto_signup1, false);

//Event listeners for login buton
login_btn=document.getElementById("login");
login_btn.addEventListener("click",login.check);
